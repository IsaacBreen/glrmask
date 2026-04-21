#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(any(test, debug_assertions))]
use std::collections::BTreeSet;
#[cfg(any(test, debug_assertions))]
use std::collections::VecDeque;
#[cfg(test)]
use std::sync::Arc;

use super::accumulator::TerminalsDisallowed;
use super::analysis::EOF;
use super::table::{Action, GLRTable};
use crate::grammar::flat::TerminalID;
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::LeveledGSS;
use smallvec::SmallVec;

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;
type GotoBatch = SmallVec<[(u32, ParserGSS); 8]>;


pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_core(table, stack.clone(), token)
}

/// Like `advance_stacks` but takes ownership of the GSS, avoiding an
/// unnecessary Arc clone when the caller doesn't need the original.
pub(crate) fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {
    advance_stacks_core(table, stack, token)
}

/// Advance the GSS by one token.
///
/// First try the deterministic single-chain path: repeatedly reduce a flat LR
/// stack, and finish immediately if that path ends in a pure shift.
///
/// If the frontier is ambiguous, or the deterministic path stops without a
/// pure shift, fall back to the GLR path: build the reduce closure to a
/// fixpoint and return the shifted next frontier.
fn advance_stacks_core(table: &GLRTable, mut gss: ParserGSS, token: TerminalID) -> ParserGSS {
    // Fast path: single state with a pure shift action (most common case).
    if let Some(state) = gss.single_exclusive_top_value() {
        if let Some(Action::Shift(target, is_replace)) = table.action(state, token) {
            return if *is_replace {
                gss.popn(1).push(*target)
            } else {
                gss.push(*target)
            };
        }
    }

    if advance_deterministically(table, &mut gss, token) {
        return gss;
    }

    advance_nondeterministically(table, gss, token)
}

fn shift_frontier(table: &GLRTable, gss: ParserGSS, token: TerminalID) -> ParserGSS {
    let mut shift_pairs = SmallVec::<[(u32, u32); 8]>::new();
    for state in gss.peek_values() {
        if let Some(target) = table.action(state, token).and_then(Action::shift_target) {
            shift_pairs.push((state, target));
        }
    }
    gss.remap_top_values_owned(shift_pairs)
}

fn apply_gotos(mut gss: ParserGSS, gotos: GotoBatch) -> ParserGSS {
    for (target, base) in gotos {
        gss = gss.absorb_push_same_acc(target, &base);
    }
    gss
}

fn add_goto(gotos: &mut GotoBatch, target: u32, base: ParserGSS) {
    if let Some((_, existing)) = gotos.iter_mut().find(|(t, _)| *t == target) {
        *existing = existing.merge(&base);
    } else {
        gotos.push((target, base));
    }
}

fn reduce_sources(gss: &ParserGSS, state: u32, rhs_len: usize) -> ReduceSources {
    gss.isolate_pop_bases(state, rhs_len as isize)
}

/// Advance an ambiguous frontier.
///
/// `closure` accumulates unshifted branches that still need GLR reduce-closure
/// processing. `shifted` accumulates branches that have already advanced past
/// the current token and therefore belong in the returned next frontier.
///
/// Each wave starts with a fresh `next` frontier. Shiftable isolated branches
/// are moved directly into `shifted`; newly reduced branches are merged into
/// `next` and become the closure for the next wave.
fn advance_nondeterministically(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
) -> ParserGSS {
    let mut shifted = ParserGSS::empty();

    loop {
        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            let Some(action) = table.action(state, token) else {
                continue;
            };
            let mut isolated = closure.isolate(Some(state));
            if advance_deterministically(table, &mut isolated, token) {
                shifted = shifted.merge(&isolated);
                continue;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace()
                    && !table.forwarded_shifts.contains(&(state, token));
                if is_replace {
                    shifted = shifted.merge(&isolated.popn(1).push(target));
                } else {
                    shifted = shifted.merge(&isolated.push(target));
                }
            }

            action.for_each_reduce(|nt, len| {
                for (goto_from, base) in reduce_sources(&closure, state, len as usize) {
                    let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
                        continue;
                    };

                    let mut branch = if is_replace {
                        base.popn(1).push(target)
                    } else {
                        base.push(target)
                    };
                    if advance_deterministically(table, &mut branch, token) {
                        shifted = shifted.merge(&branch);
                    } else {
                        next = next.merge(&branch);
                    }
                }
            });
        }

        if next.is_empty() {
            return shifted;
        }
        closure = next;
    }
}

/// Standard LR reduce loop for the deterministic case.
///
/// When the GSS frontier is a single linear chain (no ambiguity), the GSS
/// degenerates to an ordinary flat parse stack.  This applies the textbook
/// LR reduce loop directly: inspect the top state's action, pop |rhs|
/// symbols, push the goto target, repeat — until a non-reduce action is
/// reached or the chain becomes ambiguous.
///
/// If this deterministic pass ends in a pure shift, it performs that shift
/// itself and returns true to signal that the parser step is finished.
/// Otherwise it mutates `gss` and returns false so the caller can continue
/// with the nondeterministic reduce closure.
fn advance_deterministically(
    table: &GLRTable,
    gss: &mut ParserGSS,
    token: TerminalID,
) -> bool {
    let Some(mut stack) = gss.try_virtual_stack() else {
        return false; // Ambiguous frontier — skip to the general GLR path.
    };

    #[cfg(test)]
    note_vstack_hit();

    loop {
        let Some(&state) = stack.top() else {
            break;
        };

        match table.action(state, token) {
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    if rhs_len == 1 {
                        if let Some(goto_from) = stack.parent_of_top() {
                            match table.goto_target(goto_from, *nt) {
                                Some((target, false)) if stack.replace_top(target) => continue,
                                Some((target, true)) => {
                                    // Replace goto: pop current + goto_from, push target
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    *gss = ParserGSS::empty();
                                    return false;
                                }
                            }
                        }
                    }

                    // Pop |rhs| symbols and push the goto target.
                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                            *gss = ParserGSS::empty();
                            return false;
                        }
                    }
                } else {
                    // This reduce reaches or crosses the deterministic chain's
                    // floor. Finish it at the GSS level, batch the gotos, and
                    // keep going deterministically if the rebuilt frontier is
                    // still a single chain.
                    let current = stack.into_gss();
                    let popped = current.popn(rhs_len as isize);
                    let mut normal_shifts = SmallVec::<[(u32, u32); 8]>::new();
                    let mut replace_gotos = SmallVec::<[(u32, u32); 4]>::new();
                    for goto_from in popped.peek_values() {
                        if let Some((target, is_replace)) = table.goto_target(goto_from, *nt) {
                            if is_replace {
                                replace_gotos.push((goto_from, target));
                            } else {
                                normal_shifts.push((goto_from, target));
                            }
                        }
                    }
                    let rebuilt = if replace_gotos.is_empty() {
                        popped.remap_top_values_owned(normal_shifts)
                    } else {
                        let mut r = popped.remap_top_values(normal_shifts);
                        for (goto_from, target) in replace_gotos {
                            let base = popped.isolate(Some(goto_from));
                            r = r.merge(&base.popn(1).push(target));
                        }
                        r
                    };
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        *gss = rebuilt;
                        return false;
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                    *gss = stack.into_gss();
                } else {
                    *gss = stack.into_gss().push(*target);
                }
                return true;
            }
            Some(Action::Split { .. }) => {
                break;
            }
            Some(Action::Accept) => {
                break;
            }
            None => break,
        }
    }

    *gss = stack.into_gss();
    false
}

pub(crate) fn stack_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    stack.peek_values().into_iter().any(|state| table.action(state, token).is_some())
}

/// Profiled version of `advance_stacks_core`.
/// Returns (result_gss, profile) where profile contains detailed timing.
#[derive(Debug, Clone, Default)]
pub struct AdvanceProfile {
    pub pure_shift: bool,
    pub deterministic_entered: bool,
    pub deterministic_finished: bool,
    pub nondeterministic_entered: bool,
    pub vstack_len: u32,
    pub n_reduces_above_floor: u32,
    pub n_floor_crossings: u32,
    pub n_nondet_waves: u32,
    pub n_nondet_branches: u32,
    pub top_states: u32,
    pub gss_depth: u32,
    pub total_ns: u64,
    /// Arc::clone cost only (production cost)
    pub clone_ns: u64,
    /// summary() BFS traversal (profiling-only overhead)
    pub summary_ns: u64,
    pub fast_path_ns: u64,
    pub det_ns: u64,
    pub nondet_ns: u64,
    /// 0 = not entered, 1 = shift (finished), 2 = split, 3 = accept, 4 = no action, 5 = no top, 6 = vstack fail, 7 = floor cross vstack fail
    pub det_exit_reason: u32,
    pub det_exit_state: u32,
    // --- Operation counters that explain cost ---
    /// Number of action table lookups in the deterministic loop
    pub n_det_action_lookups: u32,
    /// Number of goto lookups in the deterministic path
    pub n_det_goto_lookups: u32,
    /// Number of expensive GSS-level popn operations (floor crossings)
    pub n_det_popn_ops: u32,
    /// Time spent in deterministic action table lookups
    pub det_action_lookup_ns: u64,
    /// Time spent in deterministic goto table lookups
    pub det_goto_lookup_ns: u64,
    /// Time spent in VirtualStack pop operations above the floor
    pub det_pop_ns: u64,
    /// Time spent in VirtualStack push operations above the floor
    pub det_push_ns: u64,
    /// Time spent handling deterministic floor crossings at the GSS level
    pub det_floor_cross_ns: u64,
    /// Time spent computing deterministic floor-cross reduce sources
    pub det_floor_sources_ns: u64,
    /// Time spent rebuilding the deterministic floor-cross frontier from gotos
    pub det_floor_rebuild_ns: u64,
    /// Time spent attempting to recover a VirtualStack after deterministic floor crossing
    pub det_floor_try_vstack_ns: u64,
    /// Number of reduce source enumerations in the nondeterministic path
    pub n_nondet_reduce_ops: u32,
    /// Number of GSS merge operations in the nondeterministic path
    pub n_nondet_merges: u32,
    /// Number of GSS isolate operations in the nondeterministic path
    pub n_nondet_isolates: u32,
    /// Time spent in advance_deterministically() calls inside the nondeterministic loop
    pub nondet_det_ns: u64,
    /// Time spent in GSS isolate operations in the nondeterministic path
    pub nondet_isolate_ns: u64,
    /// Time spent in GSS merge operations in the nondeterministic path
    pub nondet_merge_ns: u64,
    /// Time spent in reduce_sources (isolate_pop_bases) in the nondeterministic path
    pub nondet_reduce_sources_ns: u64,
    /// Time spent in push operations in the nondeterministic path
    pub nondet_push_ns: u64,
}

pub(crate) fn advance_stacks_profiled(
    table: &GLRTable,
    stack: &ParserGSS,
    token: TerminalID,
) -> (ParserGSS, AdvanceProfile) {
    use std::time::Instant;
    let t_total = Instant::now();
    let mut profile = AdvanceProfile::default();

    let t_clone = Instant::now();
    let mut gss = stack.clone();
    profile.clone_ns = t_clone.elapsed().as_nanos() as u64;

    // summary() is profiling-only overhead — not in the production path
    let t_summary = Instant::now();
    let summary = stack.summary();
    profile.top_states = stack.peek_values().len() as u32;
    profile.gss_depth = summary.max_depth;
    profile.summary_ns = t_summary.elapsed().as_nanos() as u64;

    // Fast path: single state with a pure shift action
    let t_fast_path = Instant::now();
    if let Some(state) = gss.single_exclusive_top_value() {
        if let Some(Action::Shift(target, is_replace)) = table.action(state, token) {
            profile.pure_shift = true;
            profile.fast_path_ns = t_fast_path.elapsed().as_nanos() as u64;
            let result = if *is_replace {
                gss.popn(1).push(*target)
            } else {
                gss.push(*target)
            };
            profile.total_ns = t_total.elapsed().as_nanos() as u64;
            return (result, profile);
        }
    }
    profile.fast_path_ns = t_fast_path.elapsed().as_nanos() as u64;

    // Try deterministic path
    let t_det = Instant::now();
    let det_result = advance_deterministically_profiled(table, &mut gss, token, &mut profile);
    profile.det_ns = t_det.elapsed().as_nanos() as u64;

    if det_result {
        profile.deterministic_finished = true;
        profile.total_ns = t_total.elapsed().as_nanos() as u64;
        return (gss, profile);
    }

    // Nondeterministic
    let t_nondet = Instant::now();
    profile.nondeterministic_entered = true;
    let result = advance_nondeterministically_profiled(table, gss, token, &mut profile);
    profile.nondet_ns = t_nondet.elapsed().as_nanos() as u64;
    profile.total_ns = t_total.elapsed().as_nanos() as u64;
    (result, profile)
}

fn advance_deterministically_profiled(
    table: &GLRTable,
    gss: &mut ParserGSS,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> bool {
    use std::time::Instant;

    let Some(mut stack) = gss.try_virtual_stack() else {
        profile.det_exit_reason = 6; // vstack fail
        return false;
    };

    profile.deterministic_entered = true;
    profile.vstack_len = stack.len() as u32;

    loop {
        let Some(&state) = stack.top() else {
            profile.det_exit_reason = 5; // no top
            break;
        };
        profile.n_det_action_lookups += 1;
        let t_action = Instant::now();
        let action = table.action(state, token);
        profile.det_action_lookup_ns += t_action.elapsed().as_nanos() as u64;
        match action {
            Some(Action::Reduce(nt, len)) => {
                let rhs_len = *len as usize;
                if rhs_len < stack.len() {
                    profile.n_reduces_above_floor += 1;
                    if rhs_len == 1 {
                        let t_parent = Instant::now();
                        let goto_from = stack.parent_of_top();
                        profile.det_pop_ns += t_parent.elapsed().as_nanos() as u64;
                        if let Some(goto_from) = goto_from {
                            profile.n_det_goto_lookups += 1;
                            let t_goto = Instant::now();
                            let goto = table.goto_target(goto_from, *nt);
                            profile.det_goto_lookup_ns += t_goto.elapsed().as_nanos() as u64;
                            match goto {
                                Some((target, false)) => {
                                    let t_replace = Instant::now();
                                    if stack.replace_top(target) {
                                        profile.det_push_ns += t_replace.elapsed().as_nanos() as u64;
                                        continue;
                                    }
                                    profile.det_push_ns += t_replace.elapsed().as_nanos() as u64;
                                }
                                Some((target, true)) => {
                                    let t_replace = Instant::now();
                                    stack.pop(2);
                                    stack.push(target);
                                    profile.det_push_ns += t_replace.elapsed().as_nanos() as u64;
                                    continue;
                                }
                                None => {
                                    *gss = ParserGSS::empty();
                                    profile.det_exit_reason = 4; // no goto
                                    return false;
                                }
                            }
                        }
                    }

                    let t_pop = Instant::now();
                    stack.pop(rhs_len);
                    profile.det_pop_ns += t_pop.elapsed().as_nanos() as u64;
                    let goto_from = *stack.top().unwrap();
                    profile.n_det_goto_lookups += 1;
                    let t_goto = Instant::now();
                    let goto = table.goto_target(goto_from, *nt);
                    profile.det_goto_lookup_ns += t_goto.elapsed().as_nanos() as u64;
                    match goto {
                        Some((target, false)) => {
                            let t_push = Instant::now();
                            stack.push(target);
                            profile.det_push_ns += t_push.elapsed().as_nanos() as u64;
                        }
                        Some((target, true)) => {
                            let t_push = Instant::now();
                            stack.replace_top(target);
                            profile.det_push_ns += t_push.elapsed().as_nanos() as u64;
                        }
                        None => {
                            *gss = ParserGSS::empty();
                            profile.det_exit_reason = 4; // no goto
                            return false;
                        }
                    }
                } else {
                    profile.n_floor_crossings += 1;
                    profile.n_det_popn_ops += 1;
                    let t_floor = Instant::now();
                    let current = stack.into_gss();
                    let t_sources = Instant::now();
                    let popped = current.popn(rhs_len as isize);
                    let mut normal_shifts = SmallVec::<[(u32, u32); 8]>::new();
                    let mut replace_gotos = SmallVec::<[(u32, u32); 4]>::new();
                    for goto_from in popped.peek_values() {
                        profile.n_det_goto_lookups += 1;
                        let t_goto = Instant::now();
                        let goto = table.goto_target(goto_from, *nt);
                        profile.det_goto_lookup_ns += t_goto.elapsed().as_nanos() as u64;
                        if let Some((target, is_replace)) = goto {
                            if is_replace {
                                replace_gotos.push((goto_from, target));
                            } else {
                                normal_shifts.push((goto_from, target));
                            }
                        }
                    }
                    profile.det_floor_sources_ns += t_sources.elapsed().as_nanos() as u64;
                    let t_rebuild = Instant::now();
                    let rebuilt = if replace_gotos.is_empty() {
                        popped.remap_top_values_owned(normal_shifts)
                    } else {
                        let mut r = popped.remap_top_values(normal_shifts);
                        for (goto_from, target) in replace_gotos {
                            let base = popped.isolate(Some(goto_from));
                            r = r.merge(&base.popn(1).push(target));
                        }
                        r
                    };
                    profile.det_floor_rebuild_ns += t_rebuild.elapsed().as_nanos() as u64;
                    profile.det_floor_cross_ns += t_floor.elapsed().as_nanos() as u64;
                    let t_try_vstack = Instant::now();
                    let Some(next_stack) = rebuilt.try_virtual_stack() else {
                        profile.det_floor_try_vstack_ns += t_try_vstack.elapsed().as_nanos() as u64;
                        *gss = rebuilt;
                        profile.det_exit_reason = 7; // floor cross vstack fail
                        return false;
                    };
                    profile.det_floor_try_vstack_ns += t_try_vstack.elapsed().as_nanos() as u64;
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                    *gss = stack.into_gss();
                } else {
                    *gss = stack.into_gss().push(*target);
                }
                profile.det_exit_reason = 1; // shift (finished)
                return true;
            }
            Some(Action::Split { .. }) => {
                profile.det_exit_reason = 2; // split
                profile.det_exit_state = state;
                break;
            }
            Some(Action::Accept) => {
                profile.det_exit_reason = 3; // accept
                profile.det_exit_state = state;
                break;
            }
            None => {
                profile.det_exit_reason = 4; // no action
                profile.det_exit_state = state;
                break;
            }
        }
    }

    *gss = stack.into_gss();
    false
}

fn advance_nondeterministically_profiled(
    table: &GLRTable,
    mut closure: ParserGSS,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> ParserGSS {
    use std::time::Instant;
    let mut shifted = ParserGSS::empty();

    loop {
        profile.n_nondet_waves += 1;
        let mut next = ParserGSS::empty();

        for state in closure.peek_values() {
            profile.n_nondet_branches += 1;
            let Some(action) = table.action(state, token) else { continue; };

            profile.n_nondet_isolates += 1;
            let t_isolate = Instant::now();
            let mut isolated = closure.isolate(Some(state));
            profile.nondet_isolate_ns += t_isolate.elapsed().as_nanos() as u64;

            let t_nd_det = Instant::now();
            let det_ok = advance_deterministically(table, &mut isolated, token);
            profile.nondet_det_ns += t_nd_det.elapsed().as_nanos() as u64;
            if det_ok {
                profile.n_nondet_merges += 1;
                let t_merge = Instant::now();
                shifted = shifted.merge(&isolated);
                profile.nondet_merge_ns += t_merge.elapsed().as_nanos() as u64;
                continue;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace()
                    && !table.forwarded_shifts.contains(&(state, token));
                profile.n_nondet_merges += 1;
                let t_push = Instant::now();
                let pushed = if is_replace {
                    isolated.popn(1).push(target)
                } else {
                    isolated.push(target)
                };
                profile.nondet_push_ns += t_push.elapsed().as_nanos() as u64;
                let t_merge = Instant::now();
                shifted = shifted.merge(&pushed);
                profile.nondet_merge_ns += t_merge.elapsed().as_nanos() as u64;
            }

            action.for_each_reduce(|nt, len| {
                let t_rs = Instant::now();
                let sources = reduce_sources(&closure, state, len as usize);
                profile.nondet_reduce_sources_ns += t_rs.elapsed().as_nanos() as u64;
                for (goto_from, base) in sources {
                    profile.n_nondet_reduce_ops += 1;
                    let Some((target, is_replace)) = table.goto_target(goto_from, nt) else { continue; };
                    let t_push = Instant::now();
                    let mut branch = if is_replace {
                        base.popn(1).push(target)
                    } else {
                        base.push(target)
                    };
                    profile.nondet_push_ns += t_push.elapsed().as_nanos() as u64;
                    profile.n_nondet_merges += 1;
                    let t_nd_det2 = Instant::now();
                    let det_ok2 = advance_deterministically(table, &mut branch, token);
                    profile.nondet_det_ns += t_nd_det2.elapsed().as_nanos() as u64;
                    let t_merge = Instant::now();
                    if det_ok2 {
                        shifted = shifted.merge(&branch);
                    } else {
                        next = next.merge(&branch);
                    }
                    profile.nondet_merge_ns += t_merge.elapsed().as_nanos() as u64;
                }
            });
        }

        if next.is_empty() { return shifted; }
        closure = next;
    }
}

/// Returns true if any terminal in the given bitset may advance the parser stack,
/// or if the parser has a Reduce/Accept action on EOF (since reductions may
/// transition to states that can then shift on future terminals).
pub(crate) fn stack_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    stack.peek_values().into_iter().any(|state| {
        table.action.get(state as usize).is_some_and(|actions| {
            actions.keys().any(|&terminal| {
                terminals.contains(terminal as usize) || terminal == EOF
            })
        })
    })
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    #[cfg(any(test, debug_assertions))]
    {
        return stacks_accept(table, &stack_vectors(stack));
    }

    #[cfg(not(any(test, debug_assertions)))]
    {
    let has_eof_action = stack
        .peek_values()
        .iter()
        .any(|&state| table.action(state, EOF).is_some());
        has_eof_action
    }
}


// ─── Test & debug infrastructure ──────────────────────────────────


#[cfg(test)]
thread_local! {
    static VSTACK_HIT_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn note_vstack_hit() {
    VSTACK_HIT_COUNT.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn take_vstack_hit_count() -> usize {
    VSTACK_HIT_COUNT.with(|count| {
        let hits = count.get();
        count.set(0);
        hits
    })
}


#[cfg(test)]
pub(crate) struct GLRParser {
    pub table: GLRTable,
    pub stack: ParserGSS,
}

#[cfg(test)]
impl GLRParser {
    pub(crate) fn new(table: GLRTable) -> Self {
        let stack = ParserGSS::from_stacks(&[(vec![0], TerminalsDisallowed::new())]);
        Self { table, stack }
    }

    pub(crate) fn step(&self, token: TerminalID) -> (Self, bool) {
        let next_stack = advance_stacks(&self.table, &self.stack, token);
        let progressed = !next_stack.is_empty();
        (
            Self {
                table: self.table.clone(),
                stack: next_stack,
            },
            progressed,
        )
    }

    pub(crate) fn valid_terminals(&self) -> Vec<TerminalID> {
        valid_terminals_for_stacks(&self.table, &self.stack)
    }
}

#[cfg(test)]
fn dedup_stacks(stacks: impl IntoIterator<Item = Vec<u32>>) -> Vec<Vec<u32>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for stack in stacks {
        if seen.insert(stack.clone()) {
            out.push(stack);
        }
    }
    out
}

#[cfg(any(test, debug_assertions))]
fn stack_vectors(stack: &ParserGSS) -> Vec<Vec<u32>> {
    stack.to_stacks().into_iter().map(|(stack, _)| stack).collect()
}

#[cfg(any(test, debug_assertions))]
fn reduce_closure_for_lookahead(
    table: &GLRTable,
    stacks: &[Vec<u32>],
    lookahead: TerminalID,
) -> Vec<Vec<u32>> {
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();

    for stack in stacks {
        if visited.insert(stack.clone()) {
            queue.push_back(stack.clone());
        }
    }

    while let Some(stack) = queue.pop_front() {
        let Some(&state) = stack.last() else {
            continue;
        };
        let Some(action) = table.action(state, lookahead) else {
            continue;
        };
        action.for_each_reduce(|nt, len| {
            let rhs_len = len as usize;
            if stack.len() < rhs_len + 1 {
                return;
            }
            let keep_len = stack.len() - rhs_len;
            let mut reduced = stack[..keep_len].to_vec();
            let Some(&goto_from) = reduced.last() else {
                return;
            };
            let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
                return;
            };
            if is_replace {
                // Replace goto: pop goto_from, push target
                reduced.pop();
                reduced.push(target);
            } else {
                reduced.push(target);
            }
            if visited.insert(reduced.clone()) {
                queue.push_back(reduced.clone());
            }
        });
    }

    visited.into_iter().collect()
}

#[cfg(test)]
fn advance_stack_vectors(
    table: &GLRTable,
    stacks: &[Vec<u32>],
    token: TerminalID,
) -> Vec<Vec<u32>> {
    let closure = reduce_closure_for_lookahead(table, stacks, token);
    let mut next = Vec::new();
    for stack in closure {
        let Some(&state) = stack.last() else {
            continue;
        };
        if let Some(target) = table.action(state, token).and_then(Action::shift_target) {
            let is_replace = table.action(state, token).is_some_and(|a| {
                a.shift_is_replace() && !table.forwarded_shifts.contains(&(state, token))
            });
            let mut shifted = stack.clone();
            if is_replace {
                shifted.pop();
            }
            shifted.push(target);
            next.push(shifted);
        }
    }
    dedup_stacks(next)
}

#[cfg(any(test, debug_assertions))]
fn stacks_accept(table: &GLRTable, stacks: &[Vec<u32>]) -> bool {
    reduce_closure_for_lookahead(table, stacks, EOF)
        .into_iter()
        .any(|stack| {
            stack.last().is_some_and(|state| {
                matches!(
                    table.action(*state, EOF),
                    Some(Action::Accept) | Some(Action::Split { accept: true, .. })
                )
            })
        })
}

#[cfg(test)]
fn valid_terminals_for_stack_vectors(
    table: &GLRTable,
    stacks: &[Vec<u32>],
) -> Vec<TerminalID> {
    (0..table.num_terminals)
        .filter(|&terminal| !advance_stack_vectors(table, stacks, terminal).is_empty())
        .collect()
}

#[cfg(test)]
pub(crate) fn valid_terminals_for_stacks(table: &GLRTable, stack: &ParserGSS) -> Vec<TerminalID> {
    valid_terminals_for_stack_vectors(table, &stack_vectors(stack))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::grammar::flat::tests::*;
    use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

    fn build_parser(gdef: &GrammarDef) -> GLRParser {
        let grammar = AnalyzedGrammar::from_grammar_def(gdef);
        let table = GLRTable::build(&grammar);
        GLRParser::new(table)
    }

    fn make_grammar(rules: Vec<Rule>, start: u32, terminals: Vec<Terminal>) -> GrammarDef {
        GrammarDef {
            rules,
            start,
            terminals,
            ..Default::default()
        }
    }

    fn with_local_forward_replace_enabled<T>(f: impl FnOnce() -> T) -> T {
        crate::compiler::glr::table::LOCAL_FORWARD_REPLACE_OVERRIDE.with(|c| {
            c.set(Some(true));
        });
        let result = f();
        crate::compiler::glr::table::LOCAL_FORWARD_REPLACE_OVERRIDE.with(|c| {
            c.set(None);
        });
        result
    }

    /// Differential test: for every token in `input`, advance both the GSS
    /// (which uses VirtualStack) and the flat reference implementation, then
    /// verify the resulting stack sets are identical.
    fn assert_advance_matches_reference(parser: &GLRParser, input: &[TerminalID]) {
        let mut gss = parser.stack.clone();
        let mut vecs = stack_vectors(&gss);
        for (i, &token) in input.iter().enumerate() {
            let gss_advanced = advance_stacks(&parser.table, &gss, token);
            let vec_advanced = advance_stack_vectors(&parser.table, &vecs, token);

            let mut gss_stacks = dedup_stacks(stack_vectors(&gss_advanced));
            gss_stacks.sort();
            let mut ref_stacks = dedup_stacks(vec_advanced.clone());
            ref_stacks.sort();

            assert_eq!(
                gss_stacks, ref_stacks,
                "Mismatch at step {i} (token {token}):\n  GSS stacks: {:?}\n  Ref stacks: {:?}",
                gss_stacks, ref_stacks
            );
            gss = gss_advanced;
            vecs = vec_advanced;
        }
    }

    fn accepts(parser: &GLRParser, input: &[TerminalID]) -> bool {
        let mut current = GLRParser {
            table: parser.table.clone(),
            stack: parser.stack.clone(),
        };
        for &token in input {
            let (next, progressed) = current.step(token);
            if !progressed {
                return false;
            }
            current = next;
        }
        stacks_finished(&current.table, &current.stack)
    }

    #[test]
    fn test_advance_stacks_preserves_accumulator_state() {
        let gdef = simple_ab_grammar();
        let grammar = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&grammar);

        let mut acc_inner = BTreeMap::new();
        acc_inner.insert(7, BTreeSet::from([11]));
        let acc = TerminalsDisallowed(Arc::new(acc_inner));
        let gss = ParserGSS::from_stacks(&[(vec![0], acc.clone())]);

        let advanced = advance_stacks(&table, &gss, 0);
        let stacks = advanced.to_stacks();

        assert_eq!(stacks.len(), 1);
        assert_eq!(stacks[0].1, acc);
    }

    #[test]
    fn test_vstack_matches_reference_raw_o9788_n10_prefix() {
        const T_A: u32 = 0;
        const T_C: u32 = 1;
        const T_B: u32 = 2;
        const T_S: u32 = 3;

        const NT_ITEM: u32 = 0;
        const NT_OPT: u32 = 1;
        const NT_REPEAT2: u32 = 2;
        const NT_REPEAT4: u32 = 3;
        const NT_REPEAT8: u32 = 4;
        const NT_REPEAT10: u32 = 5;
        const NT_START: u32 = 6;

        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: NT_ITEM,
                    rhs: vec![Symbol::Terminal(T_A), Symbol::Terminal(T_B)],
                },
                Rule {
                    lhs: NT_ITEM,
                    rhs: vec![
                        Symbol::Terminal(T_A),
                        Symbol::Nonterminal(NT_OPT),
                        Symbol::Terminal(T_B),
                    ],
                },
                Rule {
                    lhs: NT_OPT,
                    rhs: vec![Symbol::Terminal(T_C)],
                },
                Rule {
                    lhs: NT_REPEAT2,
                    rhs: vec![Symbol::Terminal(T_S), Symbol::Nonterminal(NT_ITEM)],
                },
                Rule {
                    lhs: NT_REPEAT2,
                    rhs: vec![
                        Symbol::Terminal(T_S),
                        Symbol::Nonterminal(NT_ITEM),
                        Symbol::Terminal(T_S),
                        Symbol::Nonterminal(NT_ITEM),
                    ],
                },
                Rule {
                    lhs: NT_REPEAT4,
                    rhs: vec![Symbol::Nonterminal(NT_REPEAT2)],
                },
                Rule {
                    lhs: NT_REPEAT4,
                    rhs: vec![
                        Symbol::Nonterminal(NT_REPEAT2),
                        Symbol::Nonterminal(NT_REPEAT2),
                    ],
                },
                Rule {
                    lhs: NT_REPEAT8,
                    rhs: vec![Symbol::Nonterminal(NT_REPEAT4)],
                },
                Rule {
                    lhs: NT_REPEAT8,
                    rhs: vec![
                        Symbol::Nonterminal(NT_REPEAT4),
                        Symbol::Nonterminal(NT_REPEAT4),
                    ],
                },
                Rule {
                    lhs: NT_REPEAT10,
                    rhs: vec![Symbol::Nonterminal(NT_REPEAT8)],
                },
                Rule {
                    lhs: NT_REPEAT10,
                    rhs: vec![Symbol::Nonterminal(NT_REPEAT2)],
                },
                Rule {
                    lhs: NT_REPEAT10,
                    rhs: vec![
                        Symbol::Nonterminal(NT_REPEAT8),
                        Symbol::Nonterminal(NT_REPEAT2),
                    ],
                },
                Rule {
                    lhs: NT_START,
                    rhs: vec![Symbol::Nonterminal(NT_ITEM)],
                },
                Rule {
                    lhs: NT_START,
                    rhs: vec![
                        Symbol::Nonterminal(NT_ITEM),
                        Symbol::Nonterminal(NT_REPEAT10),
                    ],
                },
            ],
            NT_START,
            vec![tdef(T_A, "a"), tdef(T_C, "c"), tdef(T_B, "b"), tdef(T_S, "s")],
        );
        let parser = build_parser(&gdef);

        let mut prefix = vec![T_A, T_B];
        for _ in 0..9 {
            prefix.extend_from_slice(&[T_S, T_A, T_B]);
        }

        assert_advance_matches_reference(&parser, &prefix);
    }

    #[test]
    fn test_parse_simple_ab() {
        let gdef = simple_ab_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[1, 0])); 
        assert!(!accepts(&parser, &[])); 
    }

    #[test]
    fn test_parse_choice() {
        let gdef = choice_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); 
        assert!(accepts(&parser, &[1])); 
        assert!(!accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[])); 
    }

    #[test]
    fn test_parse_two_nt() {
        let gdef = two_nt_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[1])); 
    }

    #[test]
    fn test_parse_ambiguous() {
        
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Nonterminal(0),
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(0),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
            vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"+".to_vec(),
                },
            ],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); 
        assert!(accepts(&parser, &[0, 1, 0])); 
        assert!(accepts(&parser, &[0, 1, 0, 1, 0])); 
        assert!(!accepts(&parser, &[1])); 
        assert!(!accepts(&parser, &[0, 1])); 
    }

    #[test]
    fn test_parse_nullable() {
        
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![],
                }, 
            ],
            0,
            vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[])); 
        assert!(accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[0, 0])); 
    }

    #[test]
    fn test_valid_terminals() {
        let gdef = simple_ab_grammar(); 
        let parser = build_parser(&gdef);
        let valid = parser.valid_terminals();
        assert!(valid.contains(&0)); 
        assert!(!valid.contains(&1)); 
    }

    #[test]
    fn test_manual_table_repro_nullable_choice_recognition_regression() {
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(2),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            0,
            vec![tdef(0, "ab"), tdef(1, "f"), tdef(2, "c")],
        );
        let parser = build_parser(&gdef);

        assert!(
            accepts(&parser, &[0, 2]),
            "manual GLR table should accept 'ab' followed by 'c' when the middle nonterminal is nullable",
        );
    }

    #[test]
    fn test_manual_table_repro_from_ebnf_explicit_nullable_choice_regression() {
        let parsed = crate::import::ebnf::parse_ebnf(
            r#"s ::= 'ab' a 'c'
a ::=
a ::= 'f'"#,
        )
        .unwrap();
        let prepared = crate::compiler::grammar::transforms::prepare_grammar_transforms_only(parsed);

        let term_id = |bytes: &[u8]| {
            prepared
                .terminals
                .iter()
                .find_map(|terminal| match terminal {
                    Terminal::Literal { id, bytes: terminal_bytes } if terminal_bytes == bytes => Some(*id),
                    _ => None,
                })
                .unwrap()
        };

        let ab = term_id(b"ab");
        let c = term_id(b"c");

        let grammar = AnalyzedGrammar::from_grammar_def(&prepared);
        let table = GLRTable::build(&grammar);
        let parser = GLRParser::new(table);

        assert!(
            accepts(&parser, &[ab, c]),
            "parser built from the exact EBNF explicit-nullable form should accept 'ab' then 'c'",
        );
    }

    #[test]
    fn test_manual_table_control_from_ebnf_optional_nullable_choice() {
        let parsed = crate::import::ebnf::parse_ebnf(
            r#"s ::= 'ab' a 'c'
    a ::= 'f'?"#,
        )
        .unwrap();
        let prepared = crate::compiler::grammar::transforms::prepare_grammar_transforms_only(parsed);

        let term_id = |bytes: &[u8]| {
            prepared
                .terminals
                .iter()
                .find_map(|terminal| match terminal {
                    Terminal::Literal { id, bytes: terminal_bytes } if terminal_bytes == bytes => Some(*id),
                    _ => None,
                })
                .unwrap()
        };

        let ab = term_id(b"ab");
        let c = term_id(b"c");

        let grammar = AnalyzedGrammar::from_grammar_def(&prepared);
        let table = GLRTable::build(&grammar);
        let parser = GLRParser::new(table);

        assert!(
            accepts(&parser, &[ab, c]),
            "parser built from the optional-syntax form should accept 'ab' then 'c'",
        );
    }

    #[test]
    fn test_manual_table_preprepare_control_from_ebnf_explicit_nullable_choice() {
        let parsed = crate::import::ebnf::parse_ebnf(
            r#"s ::= 'ab' a 'c'
a ::=
a ::= 'f'"#,
        )
        .unwrap();

        let term_id = |bytes: &[u8]| {
            parsed
                .terminals
                .iter()
                .find_map(|terminal| match terminal {
                    Terminal::Literal { id, bytes: terminal_bytes } if terminal_bytes == bytes => Some(*id),
                    _ => None,
                })
                .unwrap()
        };

        let ab = term_id(b"ab");
        let c = term_id(b"c");

        let grammar = AnalyzedGrammar::from_grammar_def(&parsed);
        let table = GLRTable::build(&grammar);
        let parser = GLRParser::new(table);

        assert!(
            accepts(&parser, &[ab, c]),
            "parser built from parsed EBNF before prepare transforms should accept 'ab' then 'c'",
        );
    }

    #[test]
    fn test_manual_table_mre_abstract_fused_terminal_rejects_split_equivalent() {
        // Smallest abstract version of the remaining JSON-schema behavior:
        // the grammar contains only a fused terminal, not the split equivalent.
        //
        // Grammar:
        //   S -> a N
        //   N -> b 'cd'
        //
        // So [a, b, cd] is accepted but [a, b, c, d] is not.
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(1)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1), Symbol::Terminal(2)] },
            ],
            0,
            vec![
                tdef(0, "a"),
                tdef(1, "b"),
                tdef(2, "cd"),
                tdef(3, "c"),
                tdef(4, "d"),
            ],
        );

        let parser = build_parser(&gdef);

        assert!(
            accepts(&parser, &[0, 1, 2]),
            "manual GLR table should accept the fused terminal form",
        );

        assert!(
            !accepts(&parser, &[0, 1, 3, 4]),
            "manual GLR table should reject the split equivalent when the grammar only includes the fused terminal",
        );
    }

    #[test]
    fn test_manual_table_mre_json_schema_split_separator_regression() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "evt": {
                    "type": "object",
                    "properties": {
                        "addListener": {"type": "string"},
                        "removeRules": {"type": "string"}
                    }
                },
                "next": {"type": "string"}
            },
            "required": ["evt", "next"],
            "additionalProperties": false
        }"#;

        let gdef = crate::import::json_schema::json_schema_to_grammar(schema).unwrap();
        let prepared = crate::compiler::grammar::transforms::prepare_grammar_transforms_only(gdef);

        let term = |bytes: &[u8]| {
            prepared
                .terminals
                .iter()
                .find_map(|t| match t {
                    Terminal::Literal { id, bytes: b } if b == bytes => Some(*id),
                    _ => None,
                })
                .unwrap()
        };

        let string_tail = prepared
            .terminals
            .iter()
            .find_map(|t| match t {
                Terminal::Expr { id, .. } => Some(*id),
                _ => None,
            })
            .unwrap();

        let fused_input = vec![term(b"{"), term(b"\""), term(b"evt\""), term(b": "), term(b"{"), term(b"}, "), term(b"\""), term(b"next\""), term(b": "), term(b"\""), string_tail, term(b"}")];
        let split_input = vec![term(b"{"), term(b"\""), term(b"evt\""), term(b": "), term(b"{"), term(b"}"), term(b", "), term(b"\""), term(b"next\""), term(b": "), term(b"\""), string_tail, term(b"}")];

        let grammar = AnalyzedGrammar::from_grammar_def(&prepared);
        let table = GLRTable::build(&grammar);
        let parser = GLRParser::new(table);

        assert!(accepts(&parser, &fused_input), "fused path should accept");
        assert!(accepts(&parser, &split_input), "split path should also accept");
    }

    fn tdef(id: u32, name: &str) -> Terminal {
        Terminal::Literal { id, bytes: name.as_bytes().to_vec() }
    }

    #[test]
    fn test_advance_stacks_uses_virtual_stack_on_reduce_then_shift() {
        // Grammar: S -> A '+' ; A -> 'i'
        // After reading 'i', the next '+' triggers a deterministic reduce
        // followed by a shift, which should go through the VirtualStack path.
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
            vec![tdef(0, "i"), tdef(1, "+")],
        );
        let parser = build_parser(&gdef);

        let after_i = advance_stacks(&parser.table, &parser.stack, 0);
        assert!(after_i.try_virtual_stack().is_some(), "single-path stack should admit VirtualStack");

        let after_i_stacks = after_i.to_stacks();
        let top_state = *after_i_stacks[0].0.last().expect("stack should have a top state");
        let plus_action = parser.table.action(top_state, 1);

        take_vstack_hit_count();
        let after_plus = advance_stacks(&parser.table, &after_i, 1);

        assert!(!after_plus.is_empty(), "reduce-then-shift path should stay alive");
        let vstack_hits = take_vstack_hit_count();
        if matches!(plus_action, Some(Action::Reduce(_, _)) | Some(Action::Split { shift: None, .. })) {
            assert!(vstack_hits > 0, "explicit reduce path should hit try_vstack_reduces");
        }
    }

    #[test]
    fn test_glr_left_recursive() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1]),       "\"b\" accepted");
        assert!(accepts(&parser, &[1, 0]),    "\"ba\" accepted");
        assert!(accepts(&parser, &[1, 0, 0]), "\"baa\" accepted");
        
        assert!(!accepts(&parser, &[0]),    "\"a\" rejected (must start with 'b')");
        assert!(!accepts(&parser, &[1, 1]), "\"bb\" rejected (two 'b's)");
    }

    #[test]
    fn test_glr_right_recursive() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1]),          "\"b\" accepted");
        assert!(accepts(&parser, &[0, 1]),       "\"ab\" accepted");
        assert!(accepts(&parser, &[0, 0, 1]),    "\"aab\" accepted");
        assert!(accepts(&parser, &[0, 0, 0, 1]), "\"aaab\" accepted");
        
        assert!(!accepts(&parser, &[0]),     "\"a\" rejected (must end in 'b')");
        assert!(!accepts(&parser, &[1, 0]),  "\"ba\" rejected");
        assert!(!accepts(&parser, &[1, 1]),  "\"bb\" rejected");
    }

    #[test]
    fn test_glr_expression_grammar() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1), Symbol::Nonterminal(1)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },                                               
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2), Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },                                               
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(0), Symbol::Terminal(4)] },    
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },                                                  
            ],
            0,
            vec![tdef(0, "i"), tdef(1, "+"), tdef(2, "*"), tdef(3, "("), tdef(4, ")")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[0]),                   "\"i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0]),             "\"i+i\" accepted");
        assert!(accepts(&parser, &[0, 2, 0]),             "\"i*i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0, 2, 0]),       "\"i+i*i\" accepted");
        assert!(accepts(&parser, &[3, 0, 1, 0, 4, 2, 0]), "\"(i+i)*i\" accepted");
        
        assert!(!accepts(&parser, &[0, 1]),       "\"i+\" rejected (incomplete)");
        assert!(!accepts(&parser, &[0, 1, 1, 0]), "\"i++i\" rejected (invalid)");
        assert!(!accepts(&parser, &[]),           "\"\" rejected (empty)");
        assert!(!accepts(&parser, &[4]),          "\")\" rejected");
        assert!(!accepts(&parser, &[3, 0]),       "\"(i\" rejected (unclosed paren)");
    }

    #[test]
    fn test_glr_reduce_reduce_conflict() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },    
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },    
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),  "\"x\" accepted despite reduce/reduce conflict");
        assert!(!accepts(&parser, &[]), "\"\" rejected");
    }

    #[test]
    fn test_glr_epsilon_ambiguity() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },  
                Rule { lhs: 1, rhs: vec![] },                     
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },  
                Rule { lhs: 2, rhs: vec![] },                     
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[]),       "\"\" accepted (A→ε, B→ε)");
        assert!(accepts(&parser, &[0]),      "\"x\" accepted (A→x,B→ε or A→ε,B→x)");
        assert!(accepts(&parser, &[0, 0]),   "\"xx\" accepted (A→x, B→x)");
        assert!(!accepts(&parser, &[0, 0, 0]), "\"xxx\" rejected");
    }

    #[test]
    fn test_glr_highly_ambiguous() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },                             
            ],
            0,
            vec![tdef(0, "a")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),       "\"a\" accepted");
        assert!(accepts(&parser, &[0, 0]),    "\"aa\" accepted");
        assert!(accepts(&parser, &[0, 0, 0]), "\"aaa\" accepted (many parse trees)");
        assert!(!accepts(&parser, &[]),       "\"\" rejected (S not nullable)");
    }

    #[test]
    fn test_glr_nullable_before_terminal() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] }, 
                Rule { lhs: 1, rhs: vec![] },                    
            ],
            0,
            vec![tdef(0, "c"), tdef(1, "d")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1, 0]), "\"dc\" accepted (A → d c)");
        assert!(accepts(&parser, &[0]),    "\"c\" accepted (A → ε c via B→ε)");
        
        assert!(!accepts(&parser, &[1]),   "\"d\" rejected (missing 'c')");
        assert!(!accepts(&parser, &[]),    "\"\" rejected (A always requires 'c')");
    }

    #[test]
    fn test_glr_ambiguous_dangling_else() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0), Symbol::Terminal(3), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(4)] }, 
            ],
            0,
            vec![tdef(0, "if"), tdef(1, "id"), tdef(2, "then"), tdef(3, "else"), tdef(4, "other")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[0, 1, 2, 0, 1, 2, 4, 3, 4]),
            "ambiguous 'if id then if id then other else other' should be accepted");
        
        assert!(accepts(&parser, &[4]),          "\"other\" accepted");
        assert!(accepts(&parser, &[0, 1, 2, 4]), "\"if id then other\" accepted");
        assert!(!accepts(&parser, &[0, 1, 2]),   "\"if id then\" rejected (incomplete)");
    }

    #[test]
    fn test_close_token_wrapper_family_remains_parseable() {
        const OPEN: u32 = 0;
        const NUM: u32 = 1;
        const COMMA: u32 = 2;
        const CLOSE: u32 = 3;

        const START: u32 = 0;
        const BODY: u32 = 1;
        const TAIL_ELEM: u32 = 2;
        const TAIL_PACK: u32 = 3;
        const FIRST_WRAP: u32 = 10;
        const WRAPPER_COUNT: usize = 24;

        let mut rules = vec![
            Rule {
                lhs: START,
                rhs: vec![
                    Symbol::Terminal(OPEN),
                    Symbol::Terminal(NUM),
                    Symbol::Nonterminal(BODY),
                    Symbol::Terminal(CLOSE),
                ],
            },
            Rule {
                lhs: BODY,
                rhs: vec![Symbol::Nonterminal(TAIL_PACK)],
            },
            Rule {
                lhs: TAIL_ELEM,
                rhs: vec![Symbol::Terminal(COMMA), Symbol::Terminal(NUM)],
            },
            Rule {
                lhs: TAIL_PACK,
                rhs: vec![Symbol::Nonterminal(TAIL_ELEM)],
            },
            Rule {
                lhs: TAIL_PACK,
                rhs: vec![
                    Symbol::Nonterminal(TAIL_ELEM),
                    Symbol::Nonterminal(TAIL_ELEM),
                ],
            },
        ];

        for i in 0..WRAPPER_COUNT {
            let wrap_nt = FIRST_WRAP + i as u32;
            rules.push(Rule {
                lhs: wrap_nt,
                rhs: vec![Symbol::Nonterminal(TAIL_PACK)],
            });
            rules.push(Rule {
                lhs: BODY,
                rhs: vec![Symbol::Nonterminal(wrap_nt)],
            });
        }

        let gdef = make_grammar(
            rules,
            START,
            vec![tdef(OPEN, "["), tdef(NUM, "n"), tdef(COMMA, ","), tdef(CLOSE, "]")],
        );
        let parser = build_parser(&gdef);

        let mut current = GLRParser {
            table: parser.table.clone(),
            stack: parser.stack.clone(),
        };
        for &token in &[OPEN, NUM, COMMA, NUM, COMMA, NUM] {
            let (next, progressed) = current.step(token);
            assert!(progressed, "prefix token {token} should progress");
            current = next;
        }

        let advanced = advance_stacks(&current.table, &current.stack, CLOSE);

        assert!(!advanced.is_empty(), "close token should remain parseable");
        assert!(
            stacks_finished(&current.table, &advanced),
            "close token should reduce the wrapper family to a finished parse"
        );
    }

    /// Differential test: GSS advance (with VirtualStack) matches the flat
    /// reference implementation for grammars with epsilon productions and
    /// nullable nonterminals that can create intermediate `empty: true` nodes.
    #[test]
    fn test_vstack_matches_reference_nullable_grammars() {
        // Grammar 1: S → A B, A → 'x' | ε, B → 'x' | ε
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
                Rule { lhs: 1, rhs: vec![] },
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },
                Rule { lhs: 2, rhs: vec![] },
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[0]);
        assert_advance_matches_reference(&parser, &[0, 0]);

        // Grammar 2: S → S S | 'a' (highly ambiguous)
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Nonterminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            ],
            0,
            vec![tdef(0, "a")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[0, 0, 0, 0, 0]);

        // Grammar 3: S → A 'c', A → 'd' | ε (nullable before terminal)
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] },
                Rule { lhs: 1, rhs: vec![] },
            ],
            0,
            vec![tdef(0, "c"), tdef(1, "d")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[1, 0]);
        assert_advance_matches_reference(&parser, &[0]);

        // Grammar 4: S → A, A → A 'a' | 'b' (left-recursive chain)
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] },
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[1, 0, 0, 0]);

        // Grammar 5: Expression grammar (deep reduce chains across floor)
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1), Symbol::Nonterminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2), Symbol::Nonterminal(2)] },
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(0), Symbol::Terminal(4)] },
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },
            ],
            0,
            vec![tdef(0, "i"), tdef(1, "+"), tdef(2, "*"), tdef(3, "("), tdef(4, ")")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[0, 1, 0, 2, 0]);
        assert_advance_matches_reference(&parser, &[3, 0, 1, 0, 4, 2, 0]);

        // Grammar 6: Reduce/reduce conflict
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(2)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert_advance_matches_reference(&parser, &[0]);

        // Grammar 7: Wrapper family (many nonterminals, deep stack)
        {
            const OPEN: u32 = 0;
            const NUM: u32 = 1;
            const COMMA: u32 = 2;
            const CLOSE: u32 = 3;
            const START: u32 = 0;
            const BODY: u32 = 1;
            const TAIL_ELEM: u32 = 2;
            const TAIL_PACK: u32 = 3;
            const FIRST_WRAP: u32 = 10;
            const WRAPPER_COUNT: usize = 24;

            let mut rules = vec![
                Rule { lhs: START, rhs: vec![Symbol::Terminal(OPEN), Symbol::Terminal(NUM), Symbol::Nonterminal(BODY), Symbol::Terminal(CLOSE)] },
                Rule { lhs: BODY, rhs: vec![Symbol::Nonterminal(TAIL_PACK)] },
                Rule { lhs: TAIL_ELEM, rhs: vec![Symbol::Terminal(COMMA), Symbol::Terminal(NUM)] },
                Rule { lhs: TAIL_PACK, rhs: vec![Symbol::Nonterminal(TAIL_ELEM)] },
                Rule { lhs: TAIL_PACK, rhs: vec![Symbol::Nonterminal(TAIL_ELEM), Symbol::Nonterminal(TAIL_ELEM)] },
            ];
            for i in 0..WRAPPER_COUNT {
                let wrap_nt = FIRST_WRAP + i as u32;
                rules.push(Rule { lhs: wrap_nt, rhs: vec![Symbol::Nonterminal(TAIL_PACK)] });
                rules.push(Rule { lhs: BODY, rhs: vec![Symbol::Nonterminal(wrap_nt)] });
            }
            let gdef = make_grammar(rules, START, vec![tdef(OPEN, "["), tdef(NUM, "n"), tdef(COMMA, ","), tdef(CLOSE, "]")]);
            let parser = build_parser(&gdef);
            assert_advance_matches_reference(&parser, &[OPEN, NUM, COMMA, NUM, COMMA, NUM, CLOSE]);
        }
    }

    /// End-to-end test proving replace shift/goto transitions actually work.
    ///
    /// Grammar: S → A B 'c', A → 'a', B → 'b'
    ///
    /// The 3-symbol production S → A B c forces:
    /// - goto(_, B) to the state with kernel {S → AB.c} is REPLACE (dot=2, no dot=1)
    /// - shift 'c' to the state with kernel {S → ABc.} is REPLACE (dot=3, no dot=1)
    ///
    /// The test asserts the table marks these transitions as replace, that
    /// parsing still accepts the valid input, and that the GSS-level and
    /// Vec-reference implementations agree.
    #[test]
    fn test_replace_shift_and_goto_end_to_end() {
        // S(0) → A(1) B(2) T(2)
        // A(1) → T(0)
        // B(2) → T(1)
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2), Symbol::Terminal(2)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(1)] },
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b"), tdef(2, "c")],
        );
        let parser = build_parser(&gdef);

        let replace_disabled = std::env::var("GLRMASK_DISABLE_REPLACE").map_or(false, |v| v == "1");
        let replace_shift_disabled = replace_disabled
            || std::env::var("GLRMASK_DISABLE_REPLACE_SHIFT").map_or(false, |v| v == "1");
        let replace_goto_disabled = replace_disabled
            || std::env::var("GLRMASK_DISABLE_REPLACE_GOTO").map_or(false, |v| v == "1");
        let mut replace_shifts = 0u32;
        let mut replace_gotos = 0u32;
        for actions_by_terminal in &parser.table.action {
            for (_, action) in actions_by_terminal {
                match action {
                    Action::Shift(_, true) => replace_shifts += 1,
                    Action::Split { shift: Some((_, true)), .. } => replace_shifts += 1,
                    _ => {}
                }
            }
        }
        for gotos_by_nt in &parser.table.goto {
            for (_, &(_, is_replace)) in gotos_by_nt {
                if is_replace { replace_gotos += 1; }
            }
        }
        if !replace_shift_disabled {
            assert!(replace_shifts > 0, "Expected at least one replace shift, found none");
        } else {
            assert_eq!(replace_shifts, 0, "Replace shifts should be 0 when disabled");
        }
        if !replace_goto_disabled {
            assert!(replace_gotos > 0, "Expected at least one replace goto, found none");
        } else {
            assert_eq!(replace_gotos, 0, "Replace gotos should be 0 when disabled");
        }

        // 2. Parse valid input: a b c  (terminals 0 1 2)
        assert!(accepts(&parser, &[0, 1, 2]));

        // 3. Reject invalid inputs.
        assert!(!accepts(&parser, &[0, 1]));
        assert!(!accepts(&parser, &[0]));
        assert!(!accepts(&parser, &[1, 0, 2]));

        // 4. Differential: GSS path matches Vec-based reference.
        assert_advance_matches_reference(&parser, &[0, 1, 2]);
    }

    #[test]
    #[ignore = "recursive grammars are excluded by the local-forward transfer_safe guard"]
    fn test_local_forward_replace_handles_chain_grammar() {
        with_local_forward_replace_enabled(|| {
            // Replicate the integration-test grammar pattern that exercises
            // local-forward-replace:
            // N0 → N1 T1        (start → item_list "$")
            // N1 → N1 N3 | N3   (item_list → item_list item | item)
            // N3 → T0 N2        (item → "a" leaf)
            // N2 → T0           (leaf → "a")
            // N2 → T0 is single-symbol, so transferring its foo-item
            // [N3 → T0 . N2] should produce a replace shift and a
            // Reduce(..., 0) somewhere in the table.
            let gdef = make_grammar(
                vec![
                    Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)] },
                    Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(3)] },
                    Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(3)] },
                    Rule { lhs: 3, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(2)] },
                    Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },
                ],
                0,
                vec![tdef(0, "a"), tdef(1, "$")],
            );

            let parser = build_parser(&gdef);

            let has_zero_reduce = parser.table.action.iter().any(|row| {
                row.values().any(|action| match action {
                    Action::Reduce(_, 0) => true,
                    Action::Split { reduces, .. } => reduces.iter().any(|&(_, len)| len == 0),
                    _ => false,
                })
            });
            assert!(
                !has_zero_reduce,
                "recursive grammars should stay on the non-forwarded path"
            );

            // "aa$" — one item with leaf
            assert!(accepts(&parser, &[0, 0, 1]));
            // "aaaa$" — two items each with leaf
            assert!(accepts(&parser, &[0, 0, 0, 0, 1]));
            // Must not accept without the closing terminal
            assert!(!accepts(&parser, &[0, 0]));
        });
    }

    #[test]
    fn test_local_forward_replace_skips_unsafe_forwarded_terminal_shift() {
        with_local_forward_replace_enabled(|| {
            // S(0) -> A(1) T(1)
            // A(1) -> T(0)
            //
            // Shifting T(0) completes A in the target kernel while the
            // source state already has the foo item [S -> .A T(1)]. This is
            // the minimal terminal-forwarding shape, but it also introduces a
            // new pop-0 completed item in the target closure, so the safety
            // guard should reject the forwarded shift.
            let gdef = make_grammar(
                vec![
                    Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)] },
                    Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
                ],
                0,
                vec![tdef(0, "a"), tdef(1, "$")],
            );
            let parser = build_parser(&gdef);

            assert!(
                parser.table.forwarded_shifts.is_empty(),
                "unsafe terminal forwarding should fall back to the ordinary shift"
            );

            assert!(accepts(&parser, &[0, 1]));
            assert!(!accepts(&parser, &[0]));
        });
    }

    /// Test that a simple linear grammar produces an optimal table with
    /// local-forward-replace: NO reductions, ALL non-accept actions are
    /// replace shifts.
    ///
    /// Grammar:
    ///   S → A '$'
    ///   A → 'a' 'a' 'a' 'a'
    ///
    /// Without replace the table has Reduce(A/4) and Reduce(S/2).
    /// With full forward-replace the parser should produce a table where
    /// every action is either a replace shift or accept — no reductions,
    /// no gotos needed.
    #[test]
    fn test_optimal_linear_grammar_no_reductions() {
        with_local_forward_replace_enabled(|| {
            // S(0) → A(1) T(1)
            // A(1) → T(0) T(0) T(0) T(0)
            let gdef = make_grammar(
                vec![
                    Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)] },
                    Rule { lhs: 1, rhs: vec![
                        Symbol::Terminal(0), Symbol::Terminal(0),
                        Symbol::Terminal(0), Symbol::Terminal(0),
                    ]},
                ],
                0,
                vec![tdef(0, "a"), tdef(1, "$")],
            );
            let parser = build_parser(&gdef);

            // Assert: no reductions in the table. All non-accept actions
            // must be replace shifts.
            for (state_id, actions) in parser.table.action.iter().enumerate() {
                for (terminal, action) in actions {
                    match action {
                        Action::Accept => {}
                        Action::Shift(_, true) => {} // replace shift — expected
                        Action::Shift(_, false) => {
                            panic!(
                                "State {state_id}: non-replace shift on terminal {terminal}"
                            );
                        }
                        Action::Reduce(nt, len) => {
                            panic!(
                                "State {state_id}: Reduce({nt}, {len}) on terminal {terminal} — expected no reductions"
                            );
                        }
                        Action::Split { .. } => {
                            panic!(
                                "State {state_id}: Split on terminal {terminal} — expected no splits"
                            );
                        }
                    }
                }
            }

            // Parser must accept valid input.
            assert!(accepts(&parser, &[0, 0, 0, 0, 1])); // "aaaa$"

            // Parser must reject invalid inputs.
            assert!(!accepts(&parser, &[0, 0, 0, 1]));       // "aaa$" — too few 'a's
            assert!(!accepts(&parser, &[0, 0, 0, 0]));       // "aaaa" — missing '$'
            assert!(!accepts(&parser, &[0, 0, 0, 0, 0, 1])); // "aaaaa$" — too many 'a's
        });
    }
}
