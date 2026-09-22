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

use range_set_blaze::RangeSetBlaze;

use crate::automata::weighted_u32::dwa::DWA;
use crate::automata::weighted_u32::minimize::reverse_hashcons_owned;
use crate::automata::weighted_u32::minimize_acyclic::{
    PointwiseClassOrder, minimize_acyclic_owned_with_pointwise_class_order,
};
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
use super::boundary_walk::boundary_accepted_tokens;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::ds::weight::{SharedTokenSet, Weight, shared_rangeset};
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
// completion), the boundary parser is a finite DAG-shaped signed action-word
// program transcribed from the closure certificate K = Id ∪ C ∪ C²:
//
// - control-depth ports Ready(k,d) per crossing-terminal-automaton vertex k
//   and depth d in 0..=max_controls_per_gap;
// - each real terminal-DWA edge k -t-> k' substitutes t's scoped signed local
//   transfer fragment once, entered from any depth and exiting to the
//   destination depth 0 with the stamped lexical weight;
// - zero-width Entry/Finish transfer fragments go from Ready(k,d) to
//   Ready(k,d+1) for d < max_controls_per_gap; nothing is added after the
//   maximum depth;
// - final weights live only on depth-0 ports (no trailing closure for
//   unrestricted admission endpoints);
// - existing resolve_negative_codes_in_nwa runs ONCE over the assembled query;
//   output pushes are preserved until then (never resolve Entry alone).
//
// General C* loops are future support for nullable/unbounded-certified cases
// and must not be used for this class: they make the program cyclic, defeat
// exact minimization, and hide unbounded silent behavior.
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
    /// Bounded-closure certificate for the supported flat class. The composer
    /// transcribes exactly this bound as depth-indexed Ready ports.
    pub closure: ClosureCertificate,
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

/// Bounded control-closure certificate for acyclic nonnullable links.
///
/// Control words in one zero-visible-terminal gap are Entry/Return events.
/// Provider stack discipline plus effective nonnullability forces every gap
/// word into return-then-enter shape `R^a E^b`: each Entry pushes a fresh
/// child-start frame that (nonnullable) cannot reach EOF-accept without
/// consuming a terminal, so no Return can follow an in-gap Entry; each Return
/// pops a pre-existing frame, so Returns precede Entries. In an acyclic
/// (DAG) link graph the live stack is always a simple component path — a
/// repeated component would be a directed link cycle — hence with maximum
/// nesting depth H (longest root-to-leaf path in edges) `a <= H` (one Return
/// per level cascading up) and `b <= H` (one Entry per level descending), for
/// at most `2H` control events per gap. The compiler transcribes this bound
/// directly as depth-indexed Ready ports instead of cyclic C* loops.
/// Link-level cycles admit unbounded stack growth and re-entry ping-pong;
/// nullable children admit accept-ready fresh frames (Entry/Return
/// ping-pong with no terminal); both keep their loud link-time declines.
/// General C* remains future support for those classes and must not be used
/// for this one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClosureCertificate {
    /// Maximum feasible control advancements per gap (2H for nesting depth H;
    /// 2 for H=1 flat links).
    pub max_controls_per_gap: u32,
    /// Maximum link nesting depth H (longest root-to-leaf path in edges).
    pub max_nesting_depth: u32,
}

/// Check the supported-class contract for bounded closure. Loud decline
/// (never silent truncation) on: link-level control cycles (unbounded stack
/// growth / re-entry ping-pong), nullable bound children (unbounded silent
/// episodes per §7.3/§7.4 can require arbitrarily many Entry/Return events in
/// one gap; single nullable words like EF would fit small depths, but the
/// certificate cannot cheaply distinguish them from unbounded unwinding or
/// silent push growth), and anything but effectively-nonnullable canonical
/// links. No-retained-controls and canonical Finish are enforced by context
/// construction and Finish instantiation.
pub(crate) fn certify_bounded_closure(
    links: &[ScopedSubgrammarLink],
) -> Result<ClosureCertificate, String> {
    for link in links {
        if link.child_start_nullable {
            return Err(format!(
                "bounded closure unsupported: child component {} is effectively nullable; silent Entry/Return episodes have no uniform bound (general C* support is future work; use the Dynamic backend)",
                link.child_component,
            ));
        }
    }
    // Longest simple-path depth from the root (component 0) over the link
    // DAG, with loud decline on any link-level cycle. Memoized DFS over
    // component ids; components unreachable from the root still contribute
    // their own longest paths (ports are shared program-wide).
    fn longest_from(
        component: u32,
        links: &[ScopedSubgrammarLink],
        memo: &mut BTreeMap<u32, u32>,
        visiting: &mut Vec<u32>,
    ) -> Result<u32, String> {
        if let Some(&cached) = memo.get(&component) {
            return Ok(cached);
        }
        if visiting.contains(&component) {
            return Err(format!(
                "bounded closure unsupported: link-level control cycle through component {component}; unbounded Entry/Return episodes need general C* support (use the Dynamic backend)",
            ));
        }
        visiting.push(component);
        let mut best = 0u32;
        for link in links.iter().filter(|link| link.parent_component == component) {
            let child_depth = longest_from(link.child_component, links, memo, visiting)?;
            best = best.max(
                child_depth
                    .checked_add(1)
                    .ok_or_else(|| "link nesting depth overflow".to_string())?,
            );
        }
        visiting.pop();
        memo.insert(component, best);
        Ok(best)
    }
    let mut memo = BTreeMap::new();
    let mut visiting = Vec::new();
    let mut max_depth = 0u32;
    let mut roots = BTreeSet::new();
    roots.insert(0u32);
    for link in links {
        roots.insert(link.parent_component);
        roots.insert(link.child_component);
    }
    for root in roots {
        max_depth = max_depth.max(longest_from(root, links, &mut memo, &mut visiting)?);
    }
    let max_controls = max_depth
        .checked_mul(2)
        .ok_or_else(|| "control-closure depth overflow".to_string())?;
    Ok(ClosureCertificate {
        max_controls_per_gap: max_controls,
        max_nesting_depth: max_depth,
    })
}

/// Build and validate the signed-link context for one flat composition
/// (single level: every link has `parent_component == 0`).
///
/// Loud declines (never silent): control-bearing components, disagreeing
/// incoming links to a shared child, noncanonical child-start/return-pop,
/// provider-unsupported slot shapes, forwarded shifts involving slots, and
/// nullable bound children (bounded-closure certificate). The packed splice
/// (rules + terminal layout) is still built by the caller for
/// grammar/follows analysis and layout pins, but no parser behavior is ever
/// derived from it.
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
    let mut ignore_terminals = Vec::with_capacity(children.len() + 1);
    ignore_terminals.push(parent.ignore_terminal);
    for child in children {
        ignore_terminals.push(child.constraint.ignore_terminal);
    }
    let links = build_segmented_parser_links(children)?;
    build_signed_link_context_from_parts(
        tables,
        ignore_terminals,
        links,
        terminal_offsets,
        num_terminals,
        global_ignores,
        unbound_slots,
    )
}

/// Build and validate the signed-link context from explicit link components:
/// intact local tables (leaves in link-component order), per-component
/// standalone ignore terminals, and the full (possibly nested) link set with
/// local slot terminals. Nested links (`parent_component != 0`) are supported
/// exactly when the link graph is acyclic and every link is effectively
/// nonnullable and canonical: the certificate then bounds each
/// zero-terminal gap by `2H` control events for nesting depth H.
///
/// Loud declines (never silent): control-bearing components, disagreeing
/// incoming links to a shared child, noncanonical child-start/return-pop,
/// provider-unsupported slot shapes, link-level control cycles, and nullable
/// bound children.
pub(crate) fn build_signed_link_context_from_parts<'a>(
    tables: Vec<&'a GLRTable>,
    ignore_terminals: Vec<Option<TerminalID>>,
    links: Vec<ScopedSubgrammarLink>,
    terminal_offsets: &[u32],
    num_terminals: u32,
    global_ignores: bool,
    unbound_slots: BTreeSet<TerminalID>,
) -> Result<SignedLinkContext<'a>, String> {
    if tables.is_empty() {
        return Err("signed link needs at least the parent component table".to_string());
    }
    if ignore_terminals.len() != tables.len() {
        return Err(format!(
            "signed link ignore-terminal count {} differs from component count {}",
            ignore_terminals.len(),
            tables.len(),
        ));
    }
    if terminal_offsets.len() != tables.len() {
        return Err(format!(
            "signed link terminal-offset count {} differs from component count {}",
            terminal_offsets.len(),
            tables.len(),
        ));
    }
    for (index, table) in tables.iter().enumerate() {
        if !table.control_terminals.is_empty() {
            return Err(format!(
                "signed static link unsupported: component {index} carries retained linker controls; classify them as controls before static reuse",
            ));
        }
    }
    for link in &links {
        if link.parent_component as usize >= tables.len()
            || link.child_component as usize >= tables.len()
        {
            return Err(format!(
                "signed link references unknown component (parent {}, child {} with {} tables)",
                link.parent_component,
                link.child_component,
                tables.len(),
            ));
        }
    }
    validate_shared_child_links(&links)?;
    let mut state_offsets = Vec::with_capacity(tables.len());
    let mut total = 0u32;
    for table in &tables {
        state_offsets.push(total);
        total = total.checked_add(table.num_states).ok_or_else(|| {
            "signed link scoped parser-state coordinate overflow".to_string()
        })?;
    }
    let parent_table = tables[0];
    let child_tables = tables[1..].to_vec();
    let closure = certify_bounded_closure(&links)?;
    let context = SignedLinkContext {
        parent_table,
        child_tables,
        links,
        terminal_offsets: terminal_offsets.to_vec(),
        num_terminals,
        state_offsets,
        total_scoped_states: total,
        global_ignores,
        ignore_terminals,
        unbound_slots,
        closure,
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

/// Link-scoped exact transfer cache prepared for the union of terminals
/// demanded by one batch of boundary shards. Ordinary terminals are
/// characterized once per owning component; Entry/Finish controls are
/// instantiated once per link. Per-shard template compilation then selects
/// from this immutable map without repeating parser-table characterization.
pub(crate) struct PreparedFragmentTransfers {
    ordinary: BTreeMap<TerminalID, TerminalCharacterization>,
    controls: BTreeMap<TerminalID, TerminalCharacterization>,
    pub entry_keys: Vec<TerminalID>,
    pub finish_keys: Vec<TerminalID>,
    pub prepare_ms: f64,
}

/// Concurrent link-scoped transfer cache for the production component
/// pipeline. Fixed Entry/Finish controls are prepared once. Ordinary terminal
/// characterization is lazy and batched by owning component on first demand;
/// separate owners use separate locks so independent shard pipelines do not
/// serialize on unrelated parser tables.
pub(crate) struct FragmentTransferCache {
    ordinary_by_owner:
        Vec<std::sync::Mutex<BTreeMap<TerminalID, TerminalCharacterization>>>,
    special: std::sync::Mutex<BTreeMap<TerminalID, TerminalCharacterization>>,
    global_ignore_identity: std::sync::Mutex<Option<TerminalCharacterization>>,
    controls: BTreeMap<TerminalID, TerminalCharacterization>,
    entry_keys: Vec<TerminalID>,
    finish_keys: Vec<TerminalID>,
    pub prepare_ms: f64,
}

impl FragmentTransferCache {
    pub(crate) fn new(context: &SignedLinkContext) -> Result<Self, String> {
        let empty = vec![false; context.num_terminals as usize];
        let prepared = prepare_fragment_transfers(context, &empty)?;
        Ok(Self {
            ordinary_by_owner: (0..context.state_offsets.len())
                .map(|_| std::sync::Mutex::new(BTreeMap::new()))
                .collect(),
            special: std::sync::Mutex::new(BTreeMap::new()),
            global_ignore_identity: std::sync::Mutex::new(None),
            controls: prepared.controls,
            entry_keys: prepared.entry_keys,
            finish_keys: prepared.finish_keys,
            prepare_ms: prepared.prepare_ms,
        })
    }
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

/// Identity that covers every component's scoped parser-state interval.
///
/// A uniformly globally erasable ignore may be skipped while the parser is
/// inside *any* component, so its identity relation must range over the whole
/// scoped state space: the union of each component's `identity_transfer`,
/// re-scoped with that component's injection, not just the owning component's
/// interval.
fn build_global_ignore_identity(
    context: &SignedLinkContext,
) -> Result<TerminalCharacterization, String> {
    let mut escapes = Vec::new();
    for component in 0..context.state_offsets.len() as u32 {
        let table = context.component_table(component)?;
        let injection = context.injection(component)?;
        let scoped = scope_characterization(&identity_transfer(table.num_states), &injection)?;
        escapes.extend(scoped.escapes);
    }
    Ok(TerminalCharacterization {
        escapes,
        reduces: Vec::new(),
        nt_escapes: Vec::new(),
        nt_rereduces: Vec::new(),
        all_nts: BTreeSet::new(),
    })
}

/// Resolve the exact scoped transfer for one emitted composed terminal.
///
/// A component-standalone ignore terminal resolves to Identity: owner-scoped
/// when the ignore is scope-dependent, or the combined global identity when
/// `context.global_ignores` certifies the ignore is uniformly globally
/// erasable. `global_ignore_identity` caches that combined relation across the
/// emitted terminals of one library build.
fn scoped_transfer_for_terminal(
    context: &SignedLinkContext,
    terminal: TerminalID,
    global_ignore_identity: &mut Option<TerminalCharacterization>,
) -> Result<TerminalCharacterization, String> {
    if context.unbound_slots.contains(&terminal) {
        return Ok(empty_transfer());
    }
    let (owner, local) = context.terminal_owner(terminal)?;
    let table = context.component_table(owner)?;
    let owner_ignore =
        context.ignore_terminals.get(owner as usize).copied().flatten() == Some(local);
    if owner_ignore {
        if !context.global_ignores {
            // Scope-dependent ignore: identity only inside the owning
            // component's interval.
            let injection = context.injection(owner)?;
            return scope_characterization(&identity_transfer(table.num_states), &injection);
        }
        // Uniformly globally erasable ignore: identity across every component
        // scope, built once per library.
        if global_ignore_identity.is_none() {
            *global_ignore_identity = Some(build_global_ignore_identity(context)?);
        }
        return Ok(global_ignore_identity
            .as_ref()
            .expect("global ignore identity was just built")
            .clone());
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
    match characterized.get(&local) {
        Some(characterization) => {
            let injection = context.injection(owner)?;
            scope_characterization(characterization, &injection)
        }
        None => {
            // No characterization entry: exact only when the terminal has no
            // parser action anywhere in the local table.
            let has_action =
                (0..table.num_states).any(|state| table.action(state, local).is_some());
            if has_action {
                return Err(format!(
                    "signed link terminal {terminal} (component {owner} local {local}) has parser actions but no characterization",
                ));
            }
            Ok(empty_transfer())
        }
    }
}

/// Test-only: format one transfer characterization's escapes/reduces.
#[cfg(test)]
fn format_characterization(transfer: &TerminalCharacterization) -> (Vec<String>, Vec<String>) {
    let escapes = transfer
        .escapes
        .iter()
        .map(|escape| format!("R{:?} P{:?}", escape.pop, escape.pushes))
        .collect::<Vec<_>>();
    let reduces = transfer
        .reduces
        .iter()
        .map(|reduce| format!("R{:?} -> nt{}", reduce.pop, reduce.nonterminal))
        .collect::<Vec<_>>();
    (escapes, reduces)
}

/// Test-only: dump the resolved scoped transfer (pop/push escapes) for each
/// emitted terminal of a shard library. Enabled by
/// `GLRMASK_DEBUG_TRANSFER_LIBRARY`.
#[cfg(test)]
fn trace_transfer_characterization(
    context: &SignedLinkContext,
    terminal: TerminalID,
    transfer: &TerminalCharacterization,
) {
    if std::env::var_os("GLRMASK_DEBUG_TRANSFER_LIBRARY").is_none() {
        return;
    }
    let owner = context.terminal_owner(terminal).ok();
    let kind = if transfer.escapes.is_empty()
        && transfer.reduces.is_empty()
        && transfer.nt_escapes.is_empty()
        && transfer.nt_rereduces.is_empty()
    {
        "empty"
    } else {
        "characterized"
    };
    let (escapes, reduces) = format_characterization(transfer);
    eprintln!(
        "[transfer-lib] global_ignores={} terminal={terminal} owner={owner:?} kind={kind} escapes={escapes:?} reduces={reduces:?}",
        context.global_ignores,
    );
}

/// Test-only: dump one Entry/Finish control fragment's pop/push escapes.
/// Enabled by `GLRMASK_DEBUG_TRANSFER_LIBRARY`.
#[cfg(test)]
fn trace_control_fragment(label: &str, transfer: &TerminalCharacterization) {
    if std::env::var_os("GLRMASK_DEBUG_TRANSFER_LIBRARY").is_none() {
        return;
    }
    let (escapes, reduces) = format_characterization(transfer);
    eprintln!("[transfer-ctl] {label} escapes={escapes:?} reduces={reduces:?}");
}

/// Test-only: enumerate bounded accepted stack words of the compiled shard
/// parser DWA, keeping only paths whose edge-weight intersection with the final
/// weight is non-empty. Labels are encoded parser states. Enabled by
/// `GLRMASK_DEBUG_BOUNDARY_WORDS`.
#[cfg(test)]
fn trace_signed_shard_parser_words(
    start_component: u32,
    parser_dwa: &DWA,
    total_scoped_states: u32,
) {
    if std::env::var_os("GLRMASK_DEBUG_BOUNDARY_WORDS").is_none() {
        return;
    }
    const MAX_LEN: usize = 6;
    const MAX_WORDS: usize = 128;
    let mut accepted = Vec::<Vec<u32>>::new();
    let mut stack = vec![(
        parser_dwa.start_state(),
        crate::ds::weight::Weight::all(),
        Vec::<u32>::new(),
    )];
    while let Some((state_id, path_weight, word)) = stack.pop() {
        let Some(state) = parser_dwa.states().get(state_id as usize) else {
            continue;
        };
        if let Some(final_weight) = state.final_weight.as_ref()
            && !path_weight.intersection(final_weight).is_empty()
        {
            accepted.push(word.clone());
            if accepted.len() >= MAX_WORDS {
                break;
            }
        }
        if word.len() >= MAX_LEN {
            continue;
        }
        for (label, target, edge_weight) in state.transitions.entries() {
            if label < 0 {
                continue;
            }
            let next_weight = path_weight.intersection(edge_weight);
            if next_weight.is_empty() {
                continue;
            }
            let mut next_word = word.clone();
            next_word.push(label as u32);
            stack.push((target, next_weight, next_word));
        }
    }
    accepted.sort();
    accepted.dedup();
    eprintln!(
        "[signed-parser-words component={start_component}] total_scoped_states={total_scoped_states} accepted_stack_words(len<={MAX_LEN})={accepted:?}"
    );
}

/// Build the scoped fragment library for one shard's emitted terminal set.
///
/// Every emitted composed terminal resolves to exactly one scoped transfer:
/// the owner's local characterization (empty relation when the terminal has
/// no parser action anywhere — verified by scan, never assumed), the Identity
/// transfer for a component-standalone ignore terminal (owner-scoped, or the
/// combined global identity when the ignore is uniformly globally erasable),
/// or the empty relation for unbound slots. A resolvable-but-missing
/// characterization is a loud error.
pub(crate) fn build_fragment_library(
    context: &SignedLinkContext,
    emitted: &[bool],
    start_component: u32,
) -> Result<FragmentLibrary, String> {
    let prepared = prepare_fragment_transfers(context, emitted)?;
    let mut library = build_fragment_library_from_prepared(
        context,
        &prepared,
        emitted,
        start_component,
    )?;
    library.templates_ms += prepared.prepare_ms;
    Ok(library)
}

/// Prepare exact ordinary/control transfers for the union demand of a link.
/// Normal terminals are grouped by owner so the existing characterization
/// engine scans each local parser table once for all demanded terminals.
pub(crate) fn prepare_fragment_transfers(
    context: &SignedLinkContext,
    emitted_union: &[bool],
) -> Result<PreparedFragmentTransfers, String> {
    let started = Instant::now();
    if emitted_union.len() < context.num_terminals as usize {
        return Err(format!(
            "signed link transfer demand has {} terminals, expected at least {}",
            emitted_union.len(), context.num_terminals,
        ));
    }
    let mut ordinary = BTreeMap::<TerminalID, TerminalCharacterization>::new();
    // Uniformly globally erasable ignores share one scoped identity covering
    // every component; build it at most once per link transfer batch.
    let mut global_ignore_identity: Option<TerminalCharacterization> = None;
    let mut ordinary_by_owner = BTreeMap::<u32, Vec<(TerminalID, TerminalID)>>::new();
    for (terminal, demanded) in emitted_union.iter().enumerate() {
        if !demanded {
            continue;
        }
        let terminal = terminal as TerminalID;
        if (terminal as usize) >= context.num_terminals as usize {
            return Err(format!(
                "signed link transfer demand emits terminal {terminal} outside the composed domain {}",
                context.num_terminals,
            ));
        }
        if context.unbound_slots.contains(&terminal) {
            ordinary.insert(terminal, empty_transfer());
            continue;
        }
        let (owner, local) = context.terminal_owner(terminal)?;
        let table = context.component_table(owner)?;
        let owner_ignore = context
            .ignore_terminals
            .get(owner as usize)
            .copied()
            .flatten()
            == Some(local);
        if owner_ignore {
            let transfer = if context.global_ignores {
                if global_ignore_identity.is_none() {
                    global_ignore_identity = Some(build_global_ignore_identity(context)?);
                }
                global_ignore_identity
                    .as_ref()
                    .expect("global ignore identity was just built")
                    .clone()
            } else {
                scope_characterization(
                    &identity_transfer(table.num_states),
                    &context.injection(owner)?,
                )?
            };
            ordinary.insert(terminal, transfer);
            continue;
        }
        if local >= table.num_terminals {
            return Err(format!(
                "signed link terminal {terminal} resolves to local {local} outside component {owner} domain {}",
                table.num_terminals,
            ));
        }
        ordinary_by_owner
            .entry(owner)
            .or_default()
            .push((terminal, local));
    }
    for (owner, terminals) in ordinary_by_owner {
        let table = context.component_table(owner)?;
        let mut selected = vec![false; table.num_terminals as usize];
        for &(_, local) in &terminals {
            selected[local as usize] = true;
        }
        let characterized = characterize_selected_terminals_for_terminal_count(
            table,
            table.num_terminals,
            &selected,
        );
        let injection = context.injection(owner)?;
        for (terminal, local) in terminals {
            let transfer = match characterized.get(&local) {
                Some(characterization) => scope_characterization(characterization, &injection)?,
                None => {
                    let has_action =
                        (0..table.num_states).any(|state| table.action(state, local).is_some());
                    if has_action {
                        return Err(format!(
                            "signed link terminal {terminal} (component {owner} local {local}) has parser actions but no characterization",
                        ));
                    }
                    empty_transfer()
                }
            };
            ordinary.insert(terminal, transfer);
        }
    }

    let mut controls = BTreeMap::<TerminalID, TerminalCharacterization>::new();
    let mut entry_keys = Vec::with_capacity(context.links.len());
    let mut finish_keys = Vec::with_capacity(context.links.len());
    for (link_index, link) in context.links.iter().enumerate() {
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
        #[cfg(test)]
        trace_control_fragment(
            &format!(
                "entry link{link_index} parent{} slot{} child_start{}",
                link.parent_component, link.slot_terminal, scoped_child_start,
            ),
            &entry.characterization,
        );
        controls.insert(entry_key, entry.characterization);
        entry_keys.push(entry_key);
        let child_table = context.component_table(link.child_component)?;
        let (finish, has_local_eof_effects) =
            instantiate_finish(child_table, link, &child_injection)?;
        #[cfg(test)]
        trace_control_fragment(
            &format!(
                "finish link{link_index} child{} return_pop{}",
                link.child_component, link.return_pop,
            ),
            &finish.characterization,
        );
        if has_local_eof_effects {
            return Err(format!(
                "signed static link unsupported: child component {} performs ordinary local EOF stack work; composing through it needs outer control-choice points",
                link.child_component,
            ));
        }
        let finish_key = finish_fragment_key(context.num_terminals, link_index)?;
        controls.insert(finish_key, finish.characterization);
        finish_keys.push(finish_key);
    }

    Ok(PreparedFragmentTransfers {
        ordinary,
        controls,
        entry_keys,
        finish_keys,
        prepare_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Compile one shard's template subset from link-scoped prepared transfers.
pub(crate) fn build_fragment_library_from_prepared(
    context: &SignedLinkContext,
    prepared: &PreparedFragmentTransfers,
    emitted: &[bool],
    start_component: u32,
) -> Result<FragmentLibrary, String> {
    let templates_started = Instant::now();
    let mut combined = prepared.controls.clone();
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
        let transfer = prepared.ordinary.get(&terminal).ok_or_else(|| {
            format!(
                "signed link shard {start_component} demands terminal {terminal} absent from prepared transfer union"
            )
        })?;
        #[cfg(test)]
        trace_transfer_characterization(context, terminal, transfer);
        combined.insert(terminal, transfer.clone());
        ordinary_terms += 1;
    }
    let templates = Templates::from_characterizations(&combined);
    let templates_ms = templates_started.elapsed().as_secs_f64() * 1000.0;
    // The bounded-DAG composer requires acyclic fragments (the engine already
    // guarantees acyclic nt re-reduction graphs by loud decline/panic; this
    // checks the compiled template NWAs themselves). A cyclic fragment would
    // reintroduce unbounded control words through the back door.
    for (&key, fragment) in &templates.by_terminal_nwa {
        if !fragment.is_acyclic() {
            return Err(format!(
                "signed link fragment {key} is cyclic; bounded flat closure cannot use it",
            ));
        }
    }
    Ok(FragmentLibrary {
        templates,
        entry_keys: prepared.entry_keys.clone(),
        finish_keys: prepared.finish_keys.clone(),
        ordinary_terms,
        templates_ms,
    })
}

/// Compile one shard's fragment library from the concurrent link cache.
/// Missing ordinary transfers are characterized in owner batches and cached
/// in scoped coordinates. No semantic merge is performed: cache hits are
/// byte-for-byte the same scoped characterization a standalone build would
/// have produced.
pub(crate) fn build_fragment_library_cached(
    context: &SignedLinkContext,
    cache: &FragmentTransferCache,
    emitted: &[bool],
    start_component: u32,
) -> Result<FragmentLibrary, String> {
    let templates_started = Instant::now();
    let mut combined = cache.controls.clone();
    let mut by_owner = BTreeMap::<u32, Vec<(TerminalID, TerminalID)>>::new();
    let mut special_terminals = Vec::<TerminalID>::new();
    let mut ordinary_terms = 0usize;
    for (terminal, &demanded) in emitted.iter().enumerate() {
        if !demanded {
            continue;
        }
        let terminal = terminal as TerminalID;
        if terminal >= context.num_terminals {
            return Err(format!(
                "signed link shard {start_component} emits terminal {terminal} outside the composed domain {}",
                context.num_terminals,
            ));
        }
        ordinary_terms += 1;
        if context.unbound_slots.contains(&terminal) {
            special_terminals.push(terminal);
            continue;
        }
        let (owner, local) = context.terminal_owner(terminal)?;
        let owner_ignore = context
            .ignore_terminals
            .get(owner as usize)
            .copied()
            .flatten()
            == Some(local);
        if owner_ignore {
            special_terminals.push(terminal);
        } else {
            by_owner.entry(owner).or_default().push((terminal, local));
        }
    }

    if !special_terminals.is_empty() {
        let mut special_cache = cache
            .special
            .lock()
            .map_err(|_| "signed link special-transfer cache poisoned".to_string())?;
        let mut global_ignore = cache
            .global_ignore_identity
            .lock()
            .map_err(|_| "signed link global-ignore cache poisoned".to_string())?;
        for terminal in special_terminals {
            if !special_cache.contains_key(&terminal) {
                let transfer =
                    scoped_transfer_for_terminal(context, terminal, &mut global_ignore)?;
                special_cache.insert(terminal, transfer);
            }
            let transfer = special_cache
                .get(&terminal)
                .expect("special transfer inserted above");
            #[cfg(test)]
            trace_transfer_characterization(context, terminal, transfer);
            combined.insert(terminal, transfer.clone());
        }
    }

    for (owner, terminals) in by_owner {
        let owner_index = owner as usize;
        let owner_cache = cache.ordinary_by_owner.get(owner_index).ok_or_else(|| {
            format!("signed link terminal owner {owner} lies outside transfer cache")
        })?;
        let mut owner_cache = owner_cache
            .lock()
            .map_err(|_| format!("signed link owner {owner} transfer cache poisoned"))?;
        let table = context.component_table(owner)?;
        let missing = terminals
            .iter()
            .filter(|(_, local)| !owner_cache.contains_key(local))
            .map(|&(_, local)| local)
            .collect::<BTreeSet<_>>();
        if !missing.is_empty() {
            let mut selected = vec![false; table.num_terminals as usize];
            for &local in &missing {
                if local >= table.num_terminals {
                    return Err(format!(
                        "signed link local terminal {local} lies outside component {owner} domain {}",
                        table.num_terminals,
                    ));
                }
                selected[local as usize] = true;
            }
            let characterized = characterize_selected_terminals_for_terminal_count(
                table,
                table.num_terminals,
                &selected,
            );
            let injection = context.injection(owner)?;
            for local in missing {
                let transfer = match characterized.get(&local) {
                    Some(characterization) => {
                        scope_characterization(characterization, &injection)?
                    }
                    None => {
                        let has_action = (0..table.num_states)
                            .any(|state| table.action(state, local).is_some());
                        if has_action {
                            return Err(format!(
                                "signed link terminal local {local} in component {owner} has parser actions but no characterization",
                            ));
                        }
                        empty_transfer()
                    }
                };
                owner_cache.insert(local, transfer);
            }
        }
        for (terminal, local) in terminals {
            let transfer = owner_cache
                .get(&local)
                .expect("owner transfer characterized or cached above");
            #[cfg(test)]
            trace_transfer_characterization(context, terminal, transfer);
            combined.insert(terminal, transfer.clone());
        }
    }

    let templates = Templates::from_characterizations(&combined);
    let templates_ms = templates_started.elapsed().as_secs_f64() * 1000.0;
    for (&key, fragment) in &templates.by_terminal_nwa {
        if !fragment.is_acyclic() {
            return Err(format!(
                "signed link fragment {key} is cyclic; bounded flat closure cannot use it",
            ));
        }
    }
    Ok(FragmentLibrary {
        templates,
        entry_keys: cache.entry_keys.clone(),
        finish_keys: cache.finish_keys.clone(),
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

/// Internal token ids carrying at least one accepted original token, as a
/// membership bitmap over the shard id_map's internal-token domain.
fn candidate_kept_internal_tokens(
    id_map: &InternalIdMap,
    candidates: &BTreeSet<u32>,
) -> Vec<bool> {
    let mut kept = vec![false; id_map.vocab_tokens.internal_to_originals.len()];
    for (internal, originals) in id_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        if originals.iter().any(|original| candidates.contains(original)) {
            kept[internal] = true;
        }
    }
    kept
}

/// Count weight cells (outer TSID ranges + inner token ranges) for profile.
fn weight_cell_count(weight: &Weight) -> (usize, usize) {
    let mut outer = 0usize;
    let mut inner = 0usize;
    for (_, tokens) in weight.raw_range_values() {
        outer += 1;
        inner += tokens.ranges().count();
    }
    (outer, inner)
}

/// Project one weight onto the shard's accepted original-token universe by
/// dropping internal tokens whose originals are all outside it. Outer TSID
/// ranges are preserved exactly (emptied ranges are dropped, adjacent equal
/// ranges merged by the remap primitive).
///
/// Test-only reference: production uses `project_weight_to_kept_in_place`.
#[cfg(test)]
fn project_weight_to_kept(weight: &Weight, kept: &[bool]) -> Weight {
    weight.remap_token_sets_preserving_tsid_ranges(|tokens| {
        project_inner_token_set(tokens, kept, None)
    })
}

/// Per-shard memo for inner token-set projection results. `kept` is fixed
/// for the shard, so the result depends only on the input set. Keyed by
/// `Arc::as_ptr` with BOTH Arcs owned per entry (prevents ABA address
/// reuse). Created once before the projection loop, dropped right after.
/// The memo OWNS an immutable borrow of the shard's `kept` mask: cross-mask
/// cache reuse is unrepresentable (a different mask needs a new memo).
/// Memory bound (structural, not a latency guarantee): at most
/// `PROJECTION_MEMO_CAP` entries, each two Arcs (refcount bumps, no deep
/// copies) plus one map node; dropped with the cache.
struct ProjectionMemo<'a> {
    kept: &'a [bool],
    map: rustc_hash::FxHashMap<usize, (SharedTokenSet, SharedTokenSet)>,
    cap: usize,
    calls: u64,
    hits: u64,
    misses: u64,
    no_drop: u64,
    rebuilds: u64,
    cap_rejections: u64,
    unchanged_weights: u64,
    remapped_weights: u64,
}

const PROJECTION_MEMO_CAP: usize = 65536;

impl<'a> ProjectionMemo<'a> {
    fn new(kept: &'a [bool]) -> Self {
        Self::with_cap(kept, PROJECTION_MEMO_CAP)
    }

    fn with_cap(kept: &'a [bool], cap: usize) -> Self {
        Self {
            kept,
            map: rustc_hash::FxHashMap::default(),
            cap,
            calls: 0,
            hits: 0,
            misses: 0,
            no_drop: 0,
            rebuilds: 0,
            cap_rejections: 0,
            unchanged_weights: 0,
            remapped_weights: 0,
        }
    }

    fn project(&mut self, tokens: &SharedTokenSet) -> SharedTokenSet {
        self.calls += 1;
        let key = std::sync::Arc::as_ptr(tokens) as usize;
        if let Some((_, result)) = self.map.get(&key) {
            self.hits += 1;
            return result.clone();
        }
        self.misses += 1;
        let result = project_inner_token_set(tokens, self.kept, Some(self));
        if self.map.len() < self.cap {
            self.map.insert(key, (tokens.clone(), result.clone()));
        } else {
            self.cap_rejections += 1;
        }
        result
    }
}

/// Exact inner-set projection body shared by the uncached path and the memo
/// miss path. `memo` (when present) only feeds the no-drop/rebuild counters;
/// the computation is identical either way.
fn project_inner_token_set(
    tokens: &SharedTokenSet,
    kept: &[bool],
    memo: Option<&mut ProjectionMemo>,
) -> SharedTokenSet {
    let mut dropped = false;
    for token in tokens.iter() {
        if !kept.get(token as usize).copied().unwrap_or(true) {
            dropped = true;
            break;
        }
    }
    if !dropped {
        if let Some(memo) = memo {
            memo.no_drop += 1;
        }
        return tokens.clone();
    }
    if let Some(memo) = memo {
        memo.rebuilds += 1;
    }
    let filtered: RangeSetBlaze<u32> = tokens
        .iter()
        .filter(|token| kept.get(*token as usize).copied().unwrap_or(true))
        .collect();
    shared_rangeset(filtered)
}

/// Project one weight through a per-shard memo. Identical output to
/// `project_weight_to_kept` for the shard's `kept`; the memo must not
/// outlive the shard it was built for.
fn project_weight_to_kept_memo(
    weight: &Weight,
    memo: &mut ProjectionMemo<'_>,
) -> Weight {
    weight.remap_token_sets_preserving_tsid_ranges(|tokens| memo.project(tokens))
}

/// In-place wrapper: if EVERY inner set projects to itself (Arc::ptr_eq),
/// the weight is already fully kept — return without remapping, cloning, or
/// reassigning. Otherwise fall through to the memoized whole-weight remap
/// (double lookups on changed weights are counted transparently in the memo
/// counters). Returns true when the weight was left untouched.
fn project_weight_to_kept_in_place(
    weight: &mut Weight,
    memo: &mut ProjectionMemo<'_>,
) -> bool {
    // One extra scan, only over inner-set Arcs (no token enumeration): the
    // memo hit path is a pointer lookup + Arc clone, so unchanged weights
    // skip the TSID-map rebuild entirely. The scan's borrow must end before
    // any reassignment, so the changed decision is recorded as a flag first.
    let mut changed = false;
    for (_, tokens) in weight.raw_range_values() {
        if !std::sync::Arc::ptr_eq(&memo.project(tokens), tokens) {
            changed = true;
            break;
        }
    }
    if !changed {
        memo.unchanged_weights += 1;
        return true;
    }
    memo.remapped_weights += 1;
    let remapped = project_weight_to_kept_memo(weight, memo);
    *weight = remapped;
    false
}

/// Compile one shard: bounded-DAG assembly, single exact negative resolution,
/// table-free positive normalization, exact minimization.
///
/// Control-depth ports Ready(k,d): every real terminal-DWA edge k -t-> k'
/// substitutes its scoped transfer fragment once, entered from ANY depth
/// (zero/one/two preceding controls) and exiting to the destination depth 0.
/// Entry/Finish fragments go zero-width from Ready(k,d) to Ready(k,d+1) for
/// d < max_controls_per_gap; nothing is added after the maximum depth. This
/// transcribes the closure certificate exactly: on well-formed stacks of the
/// supported flat class, every feasible control word in a gap has length at
/// most max_controls_per_gap, so the DAG denotes the same relation as C*.
///
/// Final weights live ONLY on depth-0 ports: admission needs no trailing
/// closure (K is reflexive — every C* accepting path's prefix through its
/// last terminal exit is already an accepting bounded path), so dropping the
/// trailing loop is clean for unrestricted admission endpoints. The fused
/// ax/ay pattern (Finish between child `a` and parent `x`) is handled by the
/// next vertex's bounded pre-terminal closure, not by trailing loops.
///
/// Negative labels are resolved exactly once over the whole query; a surviving
/// negative label afterwards is a loud error. Normalization takes only the
/// scoped parser-state count — no table, hence no table-dependent optimization
/// can consult a mismatched object.
pub(crate) fn compile_signed_shard_parser(
    context: &SignedLinkContext,
    library: &FragmentLibrary,
    shard_dwa: &DWA,
    id_map: &InternalIdMap,
    start_component: u32,
) -> Result<SignedShardOutput, String> {
    if context.closure.max_controls_per_gap == 0 && !context.links.is_empty() {
        return Err(format!(
            "signed link shard {start_component} has links but a zero control bound",
        ));
    }
    if !shard_dwa.is_acyclic() {
        return Err(format!(
            "signed link shard {start_component} lexical terminal DWA is cyclic; bounded-DAG composition needs an acyclic candidate automaton",
        ));
    }
    let depths = context.closure.max_controls_per_gap as usize + 1;
    let compose_started = Instant::now();
    let mut arena = NWA::new(0, 0);
    let mut ready = vec![u32::MAX; shard_dwa.states().len() * depths];
    let port = |ports: &[u32], vertex: usize, depth: usize| ports[vertex * depths + depth];
    for (index, state) in shard_dwa.states().iter().enumerate() {
        for depth in 0..depths {
            ready[index * depths + depth] = arena.add_state();
        }
        // Depth-0 finals only (no trailing closure for admission endpoints).
        if let Some(weight) = state.final_weight.as_ref() {
            if !weight.is_empty() {
                arena.set_final_weight(port(&ready, index, 0), weight.clone());
            }
        }
    }
    let start_index = shard_dwa.start_state() as usize;
    let start_port = ready.get(start_index * depths).copied().ok_or_else(|| {
        format!("signed link shard {start_component} has no lexical start state")
    })?;
    if start_port == u32::MAX {
        return Err(format!(
            "signed link shard {start_component} never allocated its start port"
        ));
    }
    arena.set_start_states(vec![start_port]);
    // Exact per-part contribution counters: appended template states from
    // ordinary edges vs control fragments (log-only size attribution).
    let mut ordinary_appended_states: usize = 0;
    let mut control_appended_states: usize = 0;
    // Ordinary terminal edges: one fragment clone per edge, entered from every
    // depth (shared body, single exit — sound: entries converge, the exit
    // continuation is identical, so no cross-continuation leakage), exiting to
    // the destination depth 0 with the stamped lexical weight.
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
            let target_port = ready.get(target as usize * depths).copied().ok_or_else(|| {
                format!("signed link shard {start_component} edge targets unknown state {target}")
            })?;
            let body = append_weighted_fragment(&mut arena, fragment, weight, target_port)?;
            ordinary_appended_states += fragment.states().len();
            for depth in 0..depths {
                for start in &body.start_states {
                    arena.add_epsilon(port(&ready, index, depth), *start, Weight::all());
                }
            }
        }
    }
    // Bounded zero-width control program: Entry/Finish fragments cloned per
    // (port, depth), from Ready(k,d) to Ready(k,d+1). No C* self-loops: depth
    // strictly increases, so with acyclic lexical edges and acyclic fragments
    // the assembled program is acyclic by construction (asserted below).
    let all_weight = Weight::all();
    // Diagnostic size oracle (NOT production): with
    // GLRMASK_SIGNED_SHARD_NO_CONTROLS=1 set, control fragments are omitted
    // and only ordinary transfers compose. Comparing the minimized size
    // against the full build isolates the control-closure contribution. The
    // result is not a valid shard (controls required for exactness).
    let no_controls_diagnostic = std::env::var_os("GLRMASK_SIGNED_SHARD_NO_CONTROLS").is_some();
    if !no_controls_diagnostic {
        for (index, _) in shard_dwa.states().iter().enumerate() {
            for depth in 0..depths - 1 {
                for (link_index, _) in context.links.iter().enumerate() {
                    let entry_fragment = library
                        .templates
                        .by_terminal_nwa
                        .get(&library.entry_keys[link_index])
                        .ok_or_else(|| {
                            format!("signed link entry fragment {link_index} missing from library")
                        })?;
                    let entry_body = append_weighted_fragment(
                        &mut arena,
                        entry_fragment,
                        &all_weight,
                        port(&ready, index, depth + 1),
                    )?;
                    control_appended_states += entry_fragment.states().len();
                    for start in &entry_body.start_states {
                        arena.add_epsilon(port(&ready, index, depth), *start, Weight::all());
                    }
                    let finish_fragment = library
                        .templates
                        .by_terminal_nwa
                        .get(&library.finish_keys[link_index])
                        .ok_or_else(|| {
                            format!("signed link finish fragment {link_index} missing from library")
                        })?;
                    let finish_body = append_weighted_fragment(
                        &mut arena,
                        finish_fragment,
                        &all_weight,
                        port(&ready, index, depth + 1),
                    )?;
                    control_appended_states += finish_fragment.states().len();
                    for start in &finish_body.start_states {
                        arena.add_epsilon(port(&ready, index, depth), *start, Weight::all());
                    }
                }
            }
        }
    }
    let signed_states = arena.states().len();
    let signed_transitions = arena.num_transitions();
    let compose_ms = compose_started.elapsed().as_secs_f64() * 1000.0;
    // The bounded program must be acyclic: lexical edges strictly follow the
    // acyclic candidate automaton into depth 0, controls strictly increase
    // depth, fragments were checked acyclic at library build. A cycle here
    // means the certificate's premises were wrong — decline loudly rather
    // than feeding a cyclic program to cancellation and an unminimizable
    // cyclic DWA to the runtime.
    let program_acyclic = arena.is_acyclic();
    if !program_acyclic {
        return Err(format!(
            "signed link shard {start_component} assembled a cyclic control program; bounded flat certificate violated",
        ));
    }
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
    let mut parser_dwa = normalize_weighted_parser_stack_nwa_for_parser_state_count(
        context.total_scoped_states,
        &arena,
    );
    let normalize_ms = normalize_started.elapsed().as_secs_f64() * 1000.0;
    // Exact shard-local weight projection. Every mask bit the shard can ever
    // set corresponds to an original token carried by some accepting terminal
    // path to a lexical final: parser paths substitute terminal paths and
    // stamp the same lexical weights (control fragments stamp the identity),
    // so a parser accept for t implies t is in `candidates` as computed by
    // accepted_original_tokens over the same dwa/id_map. Internal tokens whose
    // originals are all outside the candidate set therefore never contribute
    // to any (stack, tsid) query; dropping them from every edge/final weight
    // preserves all masks exactly while collapsing provably-dead distinctions
    // that otherwise bloat grouping and defeat minimization. Outer TSID ranges
    // are preserved (only emptied ranges drop); unknown internal ids default
    // to kept, so the projection can only remove proven-dead content.
    let project_started = Instant::now();
    let candidates = boundary_accepted_tokens(shard_dwa, id_map);
    let kept = candidate_kept_internal_tokens(id_map, &candidates);
    let kept_count = kept.iter().filter(|&&keep| keep).count();
    let (mut cells_outer_before, mut cells_inner_before) = (0usize, 0usize);
    for state in parser_dwa.states() {
        for (_, edge_weight) in state.transitions.values() {
            let (outer, inner) = weight_cell_count(edge_weight);
            cells_outer_before += outer;
            cells_inner_before += inner;
        }
        if let Some(final_weight) = state.final_weight.as_ref() {
            let (outer, inner) = weight_cell_count(final_weight);
            cells_outer_before += outer;
            cells_inner_before += inner;
        }
    }
    // Per-shard projection memo: `kept` is fixed for this shard, so inner
    // results depend only on the input set. Created here, dropped right
    // after the loop below. Values own both Arcs (no ABA address reuse).
    let mut memo = ProjectionMemo::new(&kept);
    for state in parser_dwa.states_mut() {
        for (_, edge_weight) in state.transitions.values_mut() {
            project_weight_to_kept_in_place(edge_weight, &mut memo);
        }
        if let Some(final_weight) = state.final_weight.as_mut() {
            project_weight_to_kept_in_place(final_weight, &mut memo);
        }
    }
    let project_ms = project_started.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some() {
        eprintln!(
            "[glrmask/profile][signed_shard_projection_memo] start_component={start_component} calls={} hits={} misses={} no_drop={} rebuilds={} entries={} cap_rejections={} unchanged_weights={} remapped_weights={}",
            memo.calls,
            memo.hits,
            memo.misses,
            memo.no_drop,
            memo.rebuilds,
            memo.map.len(),
            memo.cap_rejections,
            memo.unchanged_weights,
            memo.remapped_weights,
        );
    }
    // Exact post-normalization reduction, staged cheap-first:
    // 1. reverse structural hash-cons: merges only states with identical
    //    (final weight, ordered label->(target,weight) rows) bottom-up. A pure
    //    DAG quotient, exact for the weighted language including DEFAULT
    //    labels (a label like any other in the row signature). Near-linear.
    // 2. full acyclic partition-refinement minimization on the hash-consed
    //    input (much smaller => much cheaper than on the raw determinized
    //    graph). This is the same `minimize` the established normalize stage
    //    calls, so DEFAULT/wildcard/final-weight semantics match exactly.
    // Both stages are acyclic-only by construction (cyclic inputs return
    // unchanged), safe here because the bounded program above is asserted
    // acyclic. Both run AFTER resolution on the positive deterministic DWA,
    // so caller provenance and cancellation semantics are untouched; publish
    // re-validates positivity afterwards regardless.
    //
    // On counts: the shard's emitted *grammar terminals* (559 on selected10
    // outer) are the template-substitution granularity; the 143 *tokens* are
    // the candidate vocabulary set carried on edge weights. Both are expected;
    // neither is a duplication bug. Per-edge fragment clones cannot be shared
    // across different target continuations without cross-continuation leakage
    // (exits are contextual), and cross-source sharing rarely hits on a
    // deterministic terminal DWA — reduction is the exact collapse.
    let pre_hash_states = parser_dwa.num_states();
    let pre_hash_trans = parser_dwa.num_transitions();
    let pre_hash_acyclic = parser_dwa.is_acyclic();
    let hashcons_started = Instant::now();
    let parser_dwa = reverse_hashcons_owned(parser_dwa);
    let hashcons_ms = hashcons_started.elapsed().as_secs_f64() * 1000.0;
    let post_hash_states = parser_dwa.num_states();
    let post_hash_trans = parser_dwa.num_transitions();
    // Grouping order for greedy absorption: DescendingDomain places denser
    // partial behavior functions first. The order policy affects only
    // representation choices among already compatible classes, never the
    // accepted weighted language (documented on the enum; proven by the
    // `prepared_static_minimize_orders_are_weighted_equivalent` diagnostic);
    // the DynamicDirect differential remains the arbiter.
    // Path-conditioned minimization is deliberately NOT used: its precondition
    // (edge weights already encoding cumulative live-path domains from a
    // backward-pushed construction) is unproven for determinize_with_supports
    // output — indeed the default minimize path runs push_weights first, which
    // would be unnecessary if determinize output satisfied it.
    let minimize_started = Instant::now();
    let parser_dwa = minimize_acyclic_owned_with_pointwise_class_order(
        parser_dwa,
        PointwiseClassOrder::DescendingDomain,
    );
    let minimize_order = "descending";
    let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[glrmask/profile][signed_shard_compose] start_component={start_component} terms={} signed_states={signed_states} signed_transitions={signed_transitions} ordinary_appended={ordinary_appended_states} control_appended={control_appended_states} no_controls={no_controls_diagnostic} resolved_states={} resolved_transitions={} reverse_topo={} compose_ms={compose_ms:.3} resolve_ms={resolve_ms:.3} normalize_ms={normalize_ms:.3} candidates={} kept_internals={} cells_outer_before={} cells_inner_before={} project_ms={project_ms:.3} pre_hash_states={pre_hash_states} pre_hash_trans={pre_hash_trans} pre_hash_acyclic={pre_hash_acyclic} hashcons_ms={hashcons_ms:.3} post_hash_states={post_hash_states} post_hash_trans={post_hash_trans} minimize_order={minimize_order} minimize_ms={minimize_ms:.3} parser_states={} parser_trans={}",
        library.ordinary_terms,
        arena.states().len(),
        arena.num_transitions(),
        resolved_reverse_topo.map(|layers| layers.len()).unwrap_or(usize::MAX),
        candidates.len(),
        kept_count,
        cells_outer_before,
        cells_inner_before,
        parser_dwa.num_states(),
        parser_dwa.num_transitions(),
    );
    #[cfg(test)]
    trace_signed_shard_parser_words(start_component, &parser_dwa, context.total_scoped_states);
    Ok(SignedShardOutput {
        parser_dwa,
        templates_ms: library.templates_ms,
        compose_ms,
        resolve_ms,
        normalize_ms: normalize_ms + project_ms + hashcons_ms + minimize_ms,
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

/// Strict-static trap for every dynamic mask fallback reachable from a
/// claimed supported static path. Call at the top of each dynamic mask
/// entry point (`or_recursive_dynamic_full_walk_exact`,
/// `fill_recursive_mask_by_exact_full_walk`, unified `fill_mask_dynamic` /
/// `or_mask_dynamic_additions` / candidate additions, bounded variants).
/// Env-gated so exact dynamic compositions are unaffected; strict-static
/// tests set the var and any firing fallback panics loudly with its caller
/// name instead of contributing hidden dynamic admissions.
///
/// Genuinely dynamic constraints (`uses_dynamic_runtime`, e.g. the
/// DynamicDirect reference side of a differential running in the same
/// process) are exempt: the trap targets hidden fallbacks on claimed-static
/// paths, not legitimate dynamic evaluation. The `DynamicDirect` shard-arm
/// trap in `or_segmented_boundary_shards_mask` stays absolute: evaluating a
/// `DynamicDirect` shard while a strict-static composition is installed is
/// never legitimate.
pub(crate) fn strict_static_trap_dynamic(caller: &str) {
    if strict_static_dynamic_trap_enabled() {
        panic!(
            "GLRMASK_STRICT_STATIC_TRAP_DYNAMIC: dynamic mask fallback '{caller}' fired on a strict-static path"
        );
    }
}

/// Backend-aware trap: skips genuinely dynamic constraints, fires otherwise.
/// Prefer this at dynamic mask entry points that receive the masked state.
pub(crate) fn strict_static_trap_dynamic_for_state(caller: &str, uses_dynamic_runtime: bool) {
    if uses_dynamic_runtime {
        return;
    }
    if strict_static_dynamic_permitted() {
        return;
    }
    strict_static_trap_dynamic(caller);
}

use std::cell::Cell;

thread_local! {
    static STRICT_STATIC_PERMIT_DYNAMIC_DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn strict_static_dynamic_permitted() -> bool {
    STRICT_STATIC_PERMIT_DYNAMIC_DEPTH.with(|depth| depth.get() > 0)
}

struct StrictStaticPermitGuard;

impl StrictStaticPermitGuard {
    fn enter() -> Self {
        STRICT_STATIC_PERMIT_DYNAMIC_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for StrictStaticPermitGuard {
    fn drop(&mut self) {
        STRICT_STATIC_PERMIT_DYNAMIC_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Run `run` with legitimate dynamic evaluation permitted under the
/// strict-static trap. Use only for explicitly dynamic reference paths that
/// share a process with strict-static assertions: the authoritative
/// DynamicDirect mask branch and the env-gated dynamic/static equivalence
/// cross-check. Static-side fallbacks must never be wrapped.
pub(crate) fn permit_strict_static_dynamic<R>(run: impl FnOnce() -> R) -> R {
    let _guard = StrictStaticPermitGuard::enter();
    run()
}

pub(crate) fn eof_terminal() -> TerminalID {
    EOF
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Memo must reproduce the exact reference on every input class.
    #[test]
    fn projection_memo_matches_reference() {
        use range_set_blaze::RangeSetBlaze;
        fn set(ranges: &[(u32, u32)]) -> RangeSetBlaze<u32> {
            ranges.iter().copied().map(|(s, e)| s..=e).collect()
        }
        fn weight(entries: &[(u32, &[(u32, u32)])]) -> Weight {
            Weight::from_per_tsid_token_sets(
                entries.iter().copied().map(|(tsid, rs)| (tsid, set(rs))),
            )
        }
        fn ranges_of(w: &Weight) -> Vec<(u32, u32, Vec<(u32, u32)>)> {
            w.raw_range_values()
                .map(|(r, t)| {
                    (
                        *r.start(),
                        *r.end(),
                        t.ranges().map(|x| (*x.start(), *x.end())).collect(),
                    )
                })
                .collect()
        }
        // kept pattern: alternating keep/drop over 0..32.
        let kept: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        let cases: Vec<Vec<(u32, &[(u32, u32)])>> = vec![
            vec![(0, &[(0, 31)])],                        // mixed set
            vec![(0, &[(0, 7)]), (1, &[(0, 7)])],          // repeated shape
            vec![(0, &[(0, 31)]), (2, &[(0, 31)])],        // structurally equal Arcs
            vec![(3, &[(100, 200)])],                      // out-of-domain retained
            vec![(0, &[(u32::MAX - 4, u32::MAX)])],        // u32::MAX sparse
            vec![(0, &[])],                                // empty inner
        ];
        for (ci, entries) in cases.iter().enumerate() {
            let input = weight(entries);
            let want = project_weight_to_kept(&input, &kept);
            let mut memo = ProjectionMemo::new(&kept);
            let got = project_weight_to_kept_memo(&input, &mut memo);
            assert_eq!(ranges_of(&got), ranges_of(&want), "case {ci}");
        }
        // Repeated same Arc: second call must hit.
        {
            let input = weight(&[(0, &[(0, 31)])]);
            let mut memo = ProjectionMemo::new(&kept);
            let first = project_weight_to_kept_memo(&input, &mut memo);
            let second = project_weight_to_kept_memo(&input, &mut memo);
            assert_eq!(ranges_of(&first), ranges_of(&second));
            assert!(memo.hits >= 1, "same-Arc repeat must hit");
        }
        // All-kept / all-dropped universes.
        {
            let all_kept = vec![true; 16];
            let input = weight(&[(0, &[(0, 15)])]);
            let mut memo = ProjectionMemo::new(&all_kept);
            let got = project_weight_to_kept_memo(&input, &mut memo);
            assert_eq!(ranges_of(&got), ranges_of(&input));
            let none_kept = vec![false; 16];
            let mut memo = ProjectionMemo::new(&none_kept);
            let got = project_weight_to_kept_memo(&input, &mut memo);
            assert!(got.raw_range_values().next().is_none());
        }
        // Multi-TSID preservation: outer ranges untouched.
        {
            let input = weight(&[(0, &[(0, 31)]), (2, &[(0, 31)]), (u32::MAX, &[(0, 5)])]);
            let mut memo = ProjectionMemo::new(&kept);
            let got = project_weight_to_kept_memo(&input, &mut memo);
            let want = project_weight_to_kept(&input, &kept);
            assert_eq!(ranges_of(&got), ranges_of(&want));
            let outers: Vec<(u32, u32)> =
                got.raw_range_values().map(|(r, _)| (*r.start(), *r.end())).collect();
            assert!(outers.contains(&(u32::MAX, u32::MAX)));
        }
        // Two separate cache contexts, different kept: no cross-context reuse.
        // (Each memo owns its own kept borrow, so reuse is unrepresentable.)
        {
            let input = weight(&[(0, &[(0, 15)])]);
            let kept_a = vec![true; 16];
            let kept_b = vec![false; 16];
            let mut memo_a = ProjectionMemo::new(&kept_a);
            let mut memo_b = ProjectionMemo::new(&kept_b);
            let got_a = project_weight_to_kept_memo(&input, &mut memo_a);
            let got_b = project_weight_to_kept_memo(&input, &mut memo_b);
            assert_eq!(ranges_of(&got_a), ranges_of(&input));
            assert!(got_b.raw_range_values().next().is_none());
            assert_eq!(ranges_of(&got_a), ranges_of(&project_weight_to_kept(&input, &kept_a)));
            assert_eq!(ranges_of(&got_b), ranges_of(&project_weight_to_kept(&input, &kept_b)));
        }
        // Saturation: small-capacity constructor stops inserting at cap.
        // Distinct inner sets under ONE kept mask (the production pattern):
        // first `cap` unique sets insert, later ones reject but stay exact.
        {
            let kept_a: Vec<bool> = (0..64).map(|i| i % 2 == 0).collect();
            let mut memo = ProjectionMemo::with_cap(&kept_a, 2);
            for i in 0..8u32 {
                let w = weight(&[(0, &[(i * 8, i * 8 + 7)])]);
                let got = project_weight_to_kept_memo(&w, &mut memo);
                let want = project_weight_to_kept(&w, &kept_a);
                assert_eq!(ranges_of(&got), ranges_of(&want));
            }
            assert!(memo.cap_rejections > 0, "cap must reject inserts");
            assert!(memo.map.len() <= 2);
        }
        // In-place wrapper: no-op weights untouched, changed weights match.
        {
            // All-kept: every inner set ptr-equals through the memo → true.
            let all_kept = vec![true; 200];
            let mut memo = ProjectionMemo::new(&all_kept);
            let mut untouched = weight(&[(0, &[(0, 15)]), (2, &[(100, 110)])]);
            let before: Vec<usize> = untouched
                .raw_range_values()
                .map(|(_, t)| std::sync::Arc::as_ptr(t) as usize)
                .collect();
            assert!(
                project_weight_to_kept_in_place(&mut untouched, &mut memo),
                "all-kept weight must be a no-op",
            );
            let after: Vec<usize> = untouched
                .raw_range_values()
                .map(|(_, t)| std::sync::Arc::as_ptr(t) as usize)
                .collect();
            assert_eq!(before, after, "no-op must not reassign inner Arcs");
            assert_eq!(memo.unchanged_weights, 1);
            // Mixed multi-TSID: one dropped token forces the remap path.
            let kept_mixed: Vec<bool> =
                (0..16).map(|i| i != 5).collect();
            let mut memo = ProjectionMemo::new(&kept_mixed);
            let mut changed = weight(&[(0, &[(0, 15)]), (2, &[(0, 15)])]);
            assert!(
                !project_weight_to_kept_in_place(&mut changed, &mut memo),
                "changed weight must take the remap path",
            );
            assert_eq!(memo.remapped_weights, 1);
            assert_eq!(
                ranges_of(&changed),
                ranges_of(&project_weight_to_kept(
                    &weight(&[(0, &[(0, 15)]), (2, &[(0, 15)])]),
                    &kept_mixed,
                )),
            );
        }
    }

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
    fn strict_static_trap_fires_on_every_dynamic_fallback_entry() {
        // The trap itself is env-gated so genuine dynamic compositions are
        // unaffected; strict-static tests set the var and every dynamic mask
        // fallback panics loudly instead of contributing hidden admissions.
        // Serialized with all other process-env mutation in the test build.
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1") };
        assert!(strict_static_dynamic_trap_enabled());
        let fired = std::panic::catch_unwind(|| strict_static_trap_dynamic("test_caller"));
        unsafe { std::env::remove_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC") };
        assert!(
            fired.is_err(),
            "strict-static trap must panic on a dynamic fallback entry point"
        );
        assert!(!strict_static_dynamic_trap_enabled());
        strict_static_trap_dynamic("test_caller");
    }

    #[test]
    fn nested_nonnullable_links_certify_with_doubled_depth() {
        use crate::compiler::glr::parser::ScopedSubgrammarLink;
        // An acyclic nonnullable nested chain certifies with the 2H bound:
        // depth-2 links allow at most 4 control events per gap (R^a E^b with
        // a <= 2, b <= 2). Flat links keep the exact depth-2 certificate.
        let nested = vec![
            ScopedSubgrammarLink {
                parent_component: 0,
                slot_terminal: 3,
                child_component: 1,
                child_start: 0,
                return_pop: 1,
                child_start_nullable: false,
            },
            ScopedSubgrammarLink {
                parent_component: 1,
                slot_terminal: 4,
                child_component: 2,
                child_start: 0,
                return_pop: 1,
                child_start_nullable: false,
            },
        ];
        let certificate =
            certify_bounded_closure(&nested).expect("nonnullable nested chain must certify");
        assert_eq!(certificate.max_nesting_depth, 2);
        assert_eq!(certificate.max_controls_per_gap, 4);
        let flat = vec![nested[0]];
        let flat_certificate =
            certify_bounded_closure(&flat).expect("flat link must certify");
        assert_eq!(flat_certificate.max_nesting_depth, 1);
        assert_eq!(flat_certificate.max_controls_per_gap, 2);
        // Link-level control cycles keep their loud decline: unbounded stack
        // growth and re-entry ping-pong need general C* support.
        let cyclic = vec![
            nested[0],
            nested[1],
            ScopedSubgrammarLink {
                parent_component: 2,
                slot_terminal: 5,
                child_component: 1,
                child_start: 0,
                return_pop: 1,
                child_start_nullable: false,
            },
        ];
        let error =
            certify_bounded_closure(&cyclic).expect_err("cyclic links must decline loudly");
        assert!(
            error.contains("cycle"),
            "decline must name the cycle, got: {error}",
        );
        // Nullable links keep their loud decline.
        let mut nullable = nested;
        nullable[1].child_start_nullable = true;
        let error =
            certify_bounded_closure(&nullable).expect_err("nullable links must decline loudly");
        assert!(
            error.contains("nullable"),
            "decline must name nullability, got: {error}",
        );
    }

    #[test]
    #[ignore]
    fn composer_scaffold_declines_loudly_without_hidden_fallback() {
        assert!(assemble_boundary_transfer_query().is_err());
    }

    #[test]
    fn nested_ready_depth_four_is_load_bearing() {
        use crate::compiler::glr::parser::ScopedSubgrammarLink;
        use crate::runtime::Constraint;
        use crate::Vocab;

        fn terminal_id(constraint: &Constraint, name: &str) -> TerminalID {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        }

        fn find_local_containing(constraint: &Constraint, needle: &str) -> TerminalID {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate.contains(needle))
                .unwrap() as u32
        }

        // Walk a deterministic parser DWA with a bottom-to-top stack-state
        // word using the same transition API the runtime uses. Admission is
        // exact reachability of a non-empty final weight over non-empty
        // edge weights.
        fn dwa_admits_stack_word(dwa: &DWA, word: &[u32]) -> bool {
            let mut state = dwa.start_state();
            for &symbol in word {
                let label = symbol as i32;
                let Some((target, weight)) = dwa.states()[state as usize]
                    .transitions
                    .get_entry(&label)
                else {
                    return false;
                };
                if weight.is_empty() {
                    return false;
                }
                state = target;
            }
            dwa.states()[state as usize]
                .final_weight
                .as_ref()
                .is_some_and(|weight| !weight.is_empty())
        }

        // Enumerate accepted stack words (bottom-to-top) over the
        // deterministic parser DWA via DFS. The bounded program is acyclic,
        // so the language is finite.
        fn collect_accepted_words(dwa: &DWA, max_depth: usize) -> Vec<Vec<u32>> {
            let mut out = Vec::new();
            let mut stack: Vec<(u32, Vec<u32>)> = vec![(dwa.start_state(), Vec::new())];
            while let Some((state, word)) = stack.pop() {
                if dwa.states()[state as usize]
                    .final_weight
                    .as_ref()
                    .is_some_and(|weight| !weight.is_empty())
                {
                    out.push(word.clone());
                }
                if word.len() >= max_depth {
                    continue;
                }
                for (label, target, weight) in dwa.states()[state as usize].transitions.entries() {
                    if weight.is_empty() {
                        continue;
                    }
                    if label < 0 {
                        continue;
                    }
                    let mut next_word = word.clone();
                    next_word.push(label as u32);
                    stack.push((target, next_word));
                }
            }
            out
        }

        // Real nested fixture through real grammars: P ::= "L" SUB SUB "x",
        // M ::= "m" SUB2, G ::= "g". Depth-2 chain certifies 2H = 4.
        let vocab = Vocab::new(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"m".to_vec()),
            (5, b"g".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
            &vocab,
        )
        .unwrap();
        let mid = Constraint::from_glrm_grammar(
            r#"
                start m;
                t SUB2 ::= @token(998);
                nt m ::= "m" SUB2;
            "#,
            &vocab,
        )
        .unwrap();
        let grandchild = Constraint::from_glrm_grammar(
            r#"
                start g;
                nt g ::= "g";
            "#,
            &vocab,
        )
        .unwrap();
        let sub_p = terminal_id(&parent, "SUB");
        let sub2_m = terminal_id(&mid, "SUB2");
        let n_p = parent.table.num_terminals;
        let n_m = mid.table.num_terminals;
        let n_g = grandchild.table.num_terminals;
        let leaf_offsets = vec![0, n_p, n_p + n_m];
        let num_terminals = n_p + n_m + n_g;
        // Local id of grandchild "g" (display names quote the literal).
        let g_local = find_local_containing(&grandchild, "g");
        let g_global = leaf_offsets[2] + g_local;
        let links = vec![
            ScopedSubgrammarLink {
                parent_component: 0,
                slot_terminal: sub_p,
                child_component: 1,
                child_start: 0,
                return_pop: 1,
                child_start_nullable: false,
            },
            ScopedSubgrammarLink {
                parent_component: 1,
                slot_terminal: sub2_m,
                child_component: 2,
                child_start: 0,
                return_pop: 1,
                child_start_nullable: false,
            },
        ];
        let mut context =
            build_signed_link_context_from_parts(
                vec![&parent.table, &mid.table, &grandchild.table],
                vec![None, None, None],
                links.clone(),
                &leaf_offsets,
                num_terminals,
                false,
                BTreeSet::new(),
            )
            .expect("nested signed context");
        assert_eq!(context.closure.max_nesting_depth, 2);
        assert_eq!(context.closure.max_controls_per_gap, 4);

        // Synthetic *compiler-fixture* shard: g -> g across two lexical
        // vertices. The middle gap must chain R(G1->M1) R(M1->P) E(P->M2)
        // E(M2->G2): four control advancements before the second terminal.
        let mut shard_dwa = DWA::new(0, 1);
        let s1 = shard_dwa.add_state();
        let s2 = shard_dwa.add_state();
        shard_dwa.add_transition(0, g_global as i32, s1, Weight::all());
        shard_dwa.add_transition(s1, g_global as i32, s2, Weight::all());
        shard_dwa.set_final_weight(s2, Weight::all());
        assert!(shard_dwa.is_acyclic());

        let singleton = crate::compiler::stages::equiv_types::ManyToOneIdMap {
            original_to_internal: vec![0],
            internal_to_originals: vec![vec![0]],
            representative_original_ids: vec![0],
        };
        let id_map = InternalIdMap {
            tokenizer_states: singleton.clone(),
            vocab_tokens: singleton,
            deferred_vocab_singleton_original_ids: None,
        };

        let mut emitted = vec![false; num_terminals as usize];
        emitted[g_global as usize] = true;
        let library = build_fragment_library(&context, &emitted, 0).expect("fragment library");
        let full = compile_signed_shard_parser(&context, &library, &shard_dwa, &id_map, 0)
            .expect("depth-4 compilation must succeed");

        context.closure.max_controls_per_gap = 2;
        let library2 = build_fragment_library(&context, &emitted, 0).expect("fragment library");
        let truncated = compile_signed_shard_parser(&context, &library2, &shard_dwa, &id_map, 0)
            .expect("depth-2 compilation must succeed");

        // Exact DWA-level load-bearing proof: some stack-state word reaches
        // a final in the depth-4 program but not in the depth-2 program.
        let full_words = collect_accepted_words(&full.parser_dwa, 12);
        assert!(!full_words.is_empty(), "depth-4 program must admit something");
        let mut witness: Option<Vec<u32>> = None;
        for word in &full_words {
            if !dwa_admits_stack_word(&truncated.parser_dwa, word) {
                witness = Some(word.clone());
                break;
            }
        }
        let witness =
            witness.expect("depth-4 must admit a stack word that depth-2 truncates (R,R,E,E)");
        assert!(dwa_admits_stack_word(&full.parser_dwa, &witness));
        assert!(!dwa_admits_stack_word(&truncated.parser_dwa, &witness));
        eprintln!(
            "[nested_ready_depth_four] witness_len={} witness={:?} full_states={} trunc_states={}",
            witness.len(),
            witness,
            full.parser_dwa.num_states(),
            truncated.parser_dwa.num_states(),
        );
    }

    /// Build a two-component (parent + child) signed context where both
    /// components carry their own ignore terminal. `global` selects whether the
    /// context certifies the ignores as uniformly globally erasable.
    fn two_component_ignore_context<'a>(
        parent: &'a Constraint,
        child: &'a Constraint,
        global: bool,
    ) -> (SignedLinkContext<'a>, TerminalID) {
        let sub_p = parent
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == "SUB")
            .expect("parent SUB terminal") as TerminalID;
        let parent_terminals = parent.table.num_terminals;
        let terminal_offsets = vec![0, parent_terminals];
        let num_terminals = parent_terminals + child.table.num_terminals;
        let links = vec![ScopedSubgrammarLink {
            parent_component: 0,
            slot_terminal: sub_p,
            child_component: 1,
            child_start: 0,
            return_pop: 1,
            child_start_nullable: false,
        }];
        let context = build_signed_link_context_from_parts(
            vec![&parent.table, &child.table],
            vec![parent.ignore_terminal, child.ignore_terminal],
            links,
            &terminal_offsets,
            num_terminals,
            global,
            BTreeSet::new(),
        )
        .expect("two-component signed context");
        let child_ignore_local = child.ignore_terminal.expect("child ignore terminal");
        (context, parent_terminals + child_ignore_local)
    }

    fn ignore_test_constraints() -> (Constraint, Constraint) {
        use crate::Vocab;
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" ".to_vec()),
            (2, b"\t".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore WS;
                t WS ::= "\t"+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        (parent, child)
    }

    /// A globally erasable ignore identity must apply while the parser is
    /// inside the parent *and* inside the child.
    #[test]
    fn global_ignore_identity_spans_parent_and_child_scopes() {
        let (parent, child) = ignore_test_constraints();
        let (context, child_ignore) = two_component_ignore_context(&parent, &child, true);
        assert!(context.global_ignores);
        let mut cache = None;
        let transfer =
            scoped_transfer_for_terminal(&context, child_ignore, &mut cache).expect("transfer");
        assert!(cache.is_some(), "global identity is cached once per library");
        // Child scope: the child's own start state.
        let child_start = context.state_offsets[1];
        assert!(
            !apply_characterization(&transfer, &[child_start]).is_empty(),
            "global ignore identity must apply inside the child scope",
        );
        // Parent scope: parent state zero and the last parent state.
        assert!(
            !apply_characterization(&transfer, &[0]).is_empty(),
            "global ignore identity must apply inside the parent scope",
        );
        assert!(
            !apply_characterization(&transfer, &[parent.table.num_states - 1]).is_empty(),
            "global ignore identity must apply across the whole parent interval",
        );
    }

    /// A scope-dependent (non-global) child ignore identity must stay inside
    /// the child scope and must not apply in the parent.
    #[test]
    fn scoped_child_ignore_identity_does_not_apply_in_parent() {
        let (parent, child) = ignore_test_constraints();
        let (context, child_ignore) = two_component_ignore_context(&parent, &child, false);
        assert!(!context.global_ignores);
        let mut cache = None;
        let transfer =
            scoped_transfer_for_terminal(&context, child_ignore, &mut cache).expect("transfer");
        assert!(cache.is_none(), "scoped ignore must not build a global identity");
        let child_start = context.state_offsets[1];
        assert!(
            !apply_characterization(&transfer, &[child_start]).is_empty(),
            "scoped child ignore identity must apply inside the child scope",
        );
        assert!(
            apply_characterization(&transfer, &[0]).is_empty(),
            "scoped child ignore identity must not apply in the parent scope",
        );
    }
}
