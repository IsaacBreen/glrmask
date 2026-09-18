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

use std::collections::BTreeSet;

use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::parser::ScopedSubgrammarLink;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::stages::templates::characterize::{
    FinishEndpointPolicy, FinishTransfer, InitialEscape, InitialReduce, NtEscape, NtRereduce,
    StackMatcher, TerminalCharacterization, characterize_finish_transfer,
};
use crate::grammar::flat::TerminalID;

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
