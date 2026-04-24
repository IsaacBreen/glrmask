#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(any(test, debug_assertions))]
use std::collections::BTreeSet;
#[cfg(any(test, debug_assertions))]
use std::collections::VecDeque;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::Arc;

use super::accumulator::TerminalsDisallowed;
use super::analysis::EOF;
use super::table::{Action, GLRTable};
use crate::grammar::flat::TerminalID;
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{LeveledGSS, VirtualStack};
use smallvec::SmallVec;

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;
type GotoBatch = SmallVec<[(u32, ParserGSS); 8]>;

fn detailed_advance_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_DETAILED_ADVANCE_PROFILE")
            .map(|v| {
                let n = v.trim().to_ascii_lowercase();
                matches!(n.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}


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

fn reduce_sources_from_isolated(gss: &ParserGSS, rhs_len: usize) -> ReduceSources {
    let popped = gss.popn(rhs_len as isize);
    if popped.is_empty() {
        return SmallVec::new();
    }
    if let Some(v) = popped.single_top_value() {
        let mut result = SmallVec::new();
        result.push((v, popped));
        return result;
    }
    let top_vals = popped.peek_values();
    let mut result = SmallVec::new();
    for v in top_vals {
        result.push((v, popped.isolate(Some(v))));
    }
    result
}

fn merge_into(dst: &mut ParserGSS, branch: ParserGSS) {
    if branch.is_empty() {
        return;
    }
    if dst.is_empty() {
        *dst = branch;
    } else {
        *dst = dst.merge(&branch);
    }
}

fn advance_deterministically_from_vstack(
    table: &GLRTable,
    mut stack: VirtualStack<u32, TerminalsDisallowed>,
    token: TerminalID,
) -> (ParserGSS, bool) {
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
                                    stack.pop(2);
                                    stack.push(target);
                                    continue;
                                }
                                Some(_) | None => {
                                    return (ParserGSS::empty(), false);
                                }
                            }
                        }
                    }

                    stack.pop(rhs_len);
                    let goto_from = *stack.top().unwrap();
                    match table.goto_target(goto_from, *nt) {
                        Some((target, false)) => stack.push(target),
                        Some((target, true)) => {
                            stack.replace_top(target);
                        }
                        None => {
                            return (ParserGSS::empty(), false);
                        }
                    }
                } else {
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
                        return (rebuilt, false);
                    };
                    stack = next_stack;
                }
            }
            Some(Action::Shift(target, is_replace)) => {
                if *is_replace {
                    stack.replace_top(*target);
                } else {
                    stack.push(*target);
                }
                return (stack.into_gss(), true);
            }
            Some(Action::Split { .. }) | Some(Action::Accept) | None => break,
        }
    }

    (stack.into_gss(), false)
}

fn advance_reduce_branch(
    table: &GLRTable,
    base: ParserGSS,
    target: u32,
    is_replace: bool,
    token: TerminalID,
) -> (ParserGSS, bool) {
    if let Some(mut stack) = base.try_virtual_stack() {
        if is_replace {
            stack.replace_top(target);
        } else {
            stack.push(target);
        }
        advance_deterministically_from_vstack(table, stack, token)
    } else {
        let mut branch = if is_replace {
            base.popn(1).push(target)
        } else {
            base.push(target)
        };
        let det_ok = advance_deterministically(table, &mut branch, token);
        (branch, det_ok)
    }
}

fn accumulate_det_profile(dst: &mut AdvanceProfile, src: &AdvanceProfile) {
    dst.nondet_det_action_lookup_ns += src.det_action_lookup_ns;
    dst.nondet_det_goto_lookup_ns += src.det_goto_lookup_ns;
    dst.nondet_det_pop_ns += src.det_pop_ns;
    dst.nondet_det_push_ns += src.det_push_ns;
    dst.nondet_det_floor_cross_ns += src.det_floor_cross_ns;
    dst.nondet_det_floor_sources_ns += src.det_floor_sources_ns;
    dst.nondet_det_floor_rebuild_ns += src.det_floor_rebuild_ns;
    dst.nondet_det_floor_try_vstack_ns += src.det_floor_try_vstack_ns;
}

fn advance_reduce_branch_profiled(
    table: &GLRTable,
    base: ParserGSS,
    target: u32,
    is_replace: bool,
    token: TerminalID,
    profile: &mut AdvanceProfile,
) -> (ParserGSS, bool) {
    use std::time::Instant;

    if let Some(mut stack) = base.try_virtual_stack() {
        let t_push = Instant::now();
        if is_replace {
            stack.replace_top(target);
        } else {
            stack.push(target);
        }
        profile.nondet_push_ns += t_push.elapsed().as_nanos() as u64;

        let t_nd_det = Instant::now();
        let mut det_profile = AdvanceProfile::default();
        let mut branch = stack.into_gss();
        let det_ok = advance_deterministically_profiled(table, &mut branch, token, &mut det_profile);
        profile.nondet_det_ns += t_nd_det.elapsed().as_nanos() as u64;
        accumulate_det_profile(profile, &det_profile);
        return (branch, det_ok);
    }

    let t_push = Instant::now();
    let mut branch = if is_replace {
        base.popn(1).push(target)
    } else {
        base.push(target)
    };
    profile.nondet_push_ns += t_push.elapsed().as_nanos() as u64;

    let t_nd_det = Instant::now();
    let mut det_profile = AdvanceProfile::default();
    let det_ok = advance_deterministically_profiled(table, &mut branch, token, &mut det_profile);
    profile.nondet_det_ns += t_nd_det.elapsed().as_nanos() as u64;
    accumulate_det_profile(profile, &det_profile);
    (branch, det_ok)
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
            let reduce_base = isolated.clone();
            if advance_deterministically(table, &mut isolated, token) {
                merge_into(&mut shifted, isolated);
                continue;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace()
                    && !table.forwarded_shifts.contains(&(state, token));
                if is_replace {
                    shifted = shifted.absorb_push_same_acc(target, &isolated.popn(1));
                } else {
                    shifted = shifted.absorb_push_same_acc(target, &isolated);
                }
            }

            action.for_each_reduce(|nt, len| {
                for (goto_from, base) in reduce_sources_from_isolated(&reduce_base, len as usize) {
                    let Some((target, is_replace)) = table.goto_target(goto_from, nt) else {
                        continue;
                    };

                    let (branch, det_ok) = advance_reduce_branch(
                        table,
                        base,
                        target,
                        is_replace,
                        token,
                    );
                    if det_ok {
                        match branch.into_virtual_stack() {
                            Ok(stack) => {
                            let current = std::mem::replace(&mut shifted, ParserGSS::empty());
                                shifted = current.absorb_vstack_same_acc_owned(stack);
                            }
                            Err(branch) => {
                                merge_into(&mut shifted, branch);
                            }
                        }
                    } else {
                        merge_into(&mut next, branch);
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
                    stack.push(*target);
                    *gss = stack.into_gss();
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
    /// Time spent in deterministic action table lookups performed inside nondet_det
    pub nondet_det_action_lookup_ns: u64,
    /// Time spent in deterministic goto table lookups performed inside nondet_det
    pub nondet_det_goto_lookup_ns: u64,
    /// Time spent in deterministic virtual-stack pop operations performed inside nondet_det
    pub nondet_det_pop_ns: u64,
    /// Time spent in deterministic virtual-stack push operations performed inside nondet_det
    pub nondet_det_push_ns: u64,
    /// Time spent in deterministic floor-cross handling performed inside nondet_det
    pub nondet_det_floor_cross_ns: u64,
    /// Time spent in deterministic floor-cross source enumeration inside nondet_det
    pub nondet_det_floor_sources_ns: u64,
    /// Time spent in deterministic floor-cross frontier rebuild inside nondet_det
    pub nondet_det_floor_rebuild_ns: u64,
    /// Time spent trying to recover a virtual stack after floor crossing inside nondet_det
    pub nondet_det_floor_try_vstack_ns: u64,
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
    let detailed_profile = detailed_advance_profile_enabled();
    let t_total = Instant::now();
    let mut profile = AdvanceProfile::default();

    let t_clone = Instant::now();
    let mut gss = stack.clone();
    profile.clone_ns = t_clone.elapsed().as_nanos() as u64;

    // summary() is profiling-only overhead. Keep it behind an explicit flag so
    // default profiled commits stay close to production behavior.
    if detailed_profile {
        let t_summary = Instant::now();
        let summary = stack.summary();
        profile.top_states = stack.peek_values().len() as u32;
        profile.gss_depth = summary.max_depth;
        profile.summary_ns = t_summary.elapsed().as_nanos() as u64;
    }

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
    let det_result = if detailed_profile {
        advance_deterministically_profiled(table, &mut gss, token, &mut profile)
    } else {
        advance_deterministically(table, &mut gss, token)
    };
    profile.det_ns = t_det.elapsed().as_nanos() as u64;

    if det_result {
        profile.deterministic_finished = true;
        profile.total_ns = t_total.elapsed().as_nanos() as u64;
        return (gss, profile);
    }

    // Nondeterministic
    let t_nondet = Instant::now();
    profile.nondeterministic_entered = true;
    let result = if detailed_profile {
        advance_nondeterministically_profiled(table, gss, token, &mut profile)
    } else {
        advance_nondeterministically(table, gss, token)
    };
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
                    stack.push(*target);
                    *gss = stack.into_gss();
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
            let reduce_base = isolated.clone();

            let t_nd_det = Instant::now();
            let mut nd_det_profile = AdvanceProfile::default();
            let det_ok = advance_deterministically_profiled(
                table,
                &mut isolated,
                token,
                &mut nd_det_profile,
            );
            profile.nondet_det_ns += t_nd_det.elapsed().as_nanos() as u64;
            accumulate_det_profile(profile, &nd_det_profile);
            if det_ok {
                profile.n_nondet_merges += 1;
                let t_merge = Instant::now();
                merge_into(&mut shifted, isolated);
                profile.nondet_merge_ns += t_merge.elapsed().as_nanos() as u64;
                continue;
            }

            if let Some(target) = action.shift_target() {
                let is_replace = action.shift_is_replace()
                    && !table.forwarded_shifts.contains(&(state, token));
                profile.n_nondet_merges += 1;
                let t_push = Instant::now();
                let shift_base = if is_replace {
                    isolated.popn(1)
                } else {
                    isolated.clone()
                };
                profile.nondet_push_ns += t_push.elapsed().as_nanos() as u64;
                let t_merge = Instant::now();
                shifted = shifted.absorb_push_same_acc(target, &shift_base);
                profile.nondet_merge_ns += t_merge.elapsed().as_nanos() as u64;
            }

            action.for_each_reduce(|nt, len| {
                let t_rs = Instant::now();
                let sources = reduce_sources_from_isolated(&reduce_base, len as usize);
                profile.nondet_reduce_sources_ns += t_rs.elapsed().as_nanos() as u64;
                for (goto_from, base) in sources {
                    profile.n_nondet_reduce_ops += 1;
                    let Some((target, is_replace)) = table.goto_target(goto_from, nt) else { continue; };
                    profile.n_nondet_merges += 1;
                    let (branch, det_ok2) = advance_reduce_branch_profiled(
                        table,
                        base,
                        target,
                        is_replace,
                        token,
                        profile,
                    );
                    let t_merge = Instant::now();
                    if det_ok2 {
                        match branch.into_virtual_stack() {
                            Ok(stack) => {
                                let current = std::mem::replace(&mut shifted, ParserGSS::empty());
                                shifted = current.absorb_vstack_same_acc_owned(stack);
                            }
                            Err(branch) => {
                                merge_into(&mut shifted, branch);
                            }
                        }
                    } else {
                        merge_into(&mut next, branch);
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
                terminals.contains(terminal as usize)
            })
        })
    })
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    let has_eof_action = stack
        .peek_values()
        .iter()
        .any(|&state| table.action(state, EOF).is_some());

    has_eof_action
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
    use crate::compiler::glr::table::{ActionRow, GotoRow};
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

    fn build_manual_o1051_faithful_table_and_stack() -> (GLRTable, ParserGSS) {
        let mut action: Vec<ActionRow> = vec![Default::default(); 385];
        let mut goto: Vec<GotoRow> = vec![Default::default(); 385];

        action[142].insert(47, Action::Shift(229, true));
        action[229].insert(8, Action::Reduce(39, 1));
        goto[1].insert(39, (7, false));
        action[7].insert(
            8,
            Action::Split {
                shift: Some((384, true)),
                reduces: vec![(411, 1)],
                accept: false,
            },
        );
        goto[1].insert(411, (41, false));
        action[41].insert(8, Action::Shift(5, false));

        let table = GLRTable {
            action,
            goto,
            num_states: 385,
            num_terminals: 48,
            num_rules: 0,
            rules: Vec::new(),
            forwarded_shifts: rustc_hash::FxHashSet::default(),
        };

        let gss0 = ParserGSS::from_stacks(&[(vec![0, 1, 142], TerminalsDisallowed::new())]);
        (table, gss0)
    }

    #[test]
    fn test_profiled_advance_manual_o1051_second_advance_faithful() {
        // Faithful synthetic reproduction of the traced path on b899aa0:
        //   token 47 from [0,1,142]: Shift(229, replace)
        //   token 8 from [0,1,229]: Reduce(39,1) -> goto(1,39)=7 -> Split
        //       shift branch: Shift(384, replace) => [0,1,384]
        //       reduce branch: Reduce(411,1) -> goto(1,411)=41 -> Shift(5,false) => [0,1,41,5]
        let (table, gss0) = build_manual_o1051_faithful_table_and_stack();

        let (gss1, p1) = advance_stacks_profiled(&table, &gss0, 47);
        assert!(p1.pure_shift, "first advance should be pure shift");
        assert!(
            !p1.nondeterministic_entered,
            "first advance should stay deterministic"
        );
        let s1 = gss1.to_stacks();
        assert_eq!(s1.len(), 1);
        assert_eq!(s1[0].0, vec![0, 1, 229]);

        let (gss2, p2) = advance_stacks_profiled(&table, &gss1, 8);
        assert!(p2.nondeterministic_entered, "second advance should enter nondet path");

        let mut stacks2: Vec<Vec<u32>> = gss2.to_stacks().into_iter().map(|(s, _)| s).collect();
        stacks2.sort();
        assert_eq!(stacks2, vec![vec![0, 1, 41, 5], vec![0, 1, 384]]);
    }

    #[test]
    fn test_profiled_advance_manual_o1051_second_advance_faithful_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        let (table, gss0) = build_manual_o1051_faithful_table_and_stack();
        let (gss1_base, p1) = advance_stacks_profiled(&table, &gss0, 47);
        assert!(p1.pure_shift);

        let warmup = 2_000usize;
        let warm_iters = 20_000usize;
        let cold_iters = 400usize;

        for _ in 0..warmup {
            let _ = advance_stacks_profiled(&table, &gss1_base, 8);
        }

        let mut warm_wall_ns: u128 = 0;
        let mut warm_total_ns: u128 = 0;
        let mut warm_det_ns: u128 = 0;
        let mut warm_nondet_ns: u128 = 0;

        for _ in 0..warm_iters {
            let t0 = Instant::now();
            let (gss2, p2) = advance_stacks_profiled(&table, &gss1_base, 8);
            warm_wall_ns += t0.elapsed().as_nanos();
            warm_total_ns += p2.total_ns as u128;
            warm_det_ns += p2.det_ns as u128;
            warm_nondet_ns += p2.nondet_ns as u128;

            let mut stacks2: Vec<Vec<u32>> = gss2.to_stacks().into_iter().map(|(s, _)| s).collect();
            stacks2.sort();
            assert_eq!(stacks2, vec![vec![0, 1, 41, 5], vec![0, 1, 384]]);
            assert!(p2.nondeterministic_entered);
        }

        // Cold approximation: touch a large buffer before each sample to evict
        // parser data from CPU caches.
        let mut cache_thrash = vec![0u8; 32 * 1024 * 1024];
        let mut thrash_checksum: u64 = 0;
        let mut cold_wall_ns: u128 = 0;
        let mut cold_total_ns: u128 = 0;
        let mut cold_det_ns: u128 = 0;
        let mut cold_nondet_ns: u128 = 0;

        for iter in 0..cold_iters {
            let salt = (iter as u8).wrapping_mul(17).wrapping_add(3);
            for i in (0..cache_thrash.len()).step_by(64) {
                cache_thrash[i] = cache_thrash[i].wrapping_add(salt);
                thrash_checksum = thrash_checksum.wrapping_add(cache_thrash[i] as u64);
            }

            let t0 = Instant::now();
            let (gss2, p2) = advance_stacks_profiled(&table, &gss1_base, 8);
            cold_wall_ns += t0.elapsed().as_nanos();
            cold_total_ns += p2.total_ns as u128;
            cold_det_ns += p2.det_ns as u128;
            cold_nondet_ns += p2.nondet_ns as u128;

            let mut stacks2: Vec<Vec<u32>> = gss2.to_stacks().into_iter().map(|(s, _)| s).collect();
            stacks2.sort();
            assert_eq!(stacks2, vec![vec![0, 1, 41, 5], vec![0, 1, 384]]);
            assert!(p2.nondeterministic_entered);
        }

        black_box(thrash_checksum);

        let warm_inv = 1.0 / warm_iters as f64;
        let cold_inv = 1.0 / cold_iters as f64;
        eprintln!(
            "[profiled_advance_manual_o1051_faithful][warm] avg_wall_us={:.3} avg_total_us={:.3} avg_det_us={:.3} avg_nondet_us={:.3}",
            (warm_wall_ns as f64) * warm_inv / 1_000.0,
            (warm_total_ns as f64) * warm_inv / 1_000.0,
            (warm_det_ns as f64) * warm_inv / 1_000.0,
            (warm_nondet_ns as f64) * warm_inv / 1_000.0,
        );
        eprintln!(
            "[profiled_advance_manual_o1051_faithful][cold] avg_wall_us={:.3} avg_total_us={:.3} avg_det_us={:.3} avg_nondet_us={:.3} cold_method=thrash_32MiB_stride64",
            (cold_wall_ns as f64) * cold_inv / 1_000.0,
            (cold_total_ns as f64) * cold_inv / 1_000.0,
            (cold_det_ns as f64) * cold_inv / 1_000.0,
            (cold_nondet_ns as f64) * cold_inv / 1_000.0,
        );
    }

    #[test]
    fn test_advance_manual_o1051_second_advance_faithful_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        let (table, gss0) = build_manual_o1051_faithful_table_and_stack();
        let gss1_base = advance_stacks(&table, &gss0, 47);

        let warmup = 2_000usize;
        let warm_iters = 50_000usize;
        let cold_iters = 1_000usize;

        for _ in 0..warmup {
            let _ = advance_stacks(&table, &gss1_base, 8);
        }

        let mut warm_wall_ns: u128 = 0;
        for _ in 0..warm_iters {
            let t0 = Instant::now();
            let gss2 = advance_stacks(&table, &gss1_base, 8);
            warm_wall_ns += t0.elapsed().as_nanos();

            let mut stacks2: Vec<Vec<u32>> = gss2.to_stacks().into_iter().map(|(s, _)| s).collect();
            stacks2.sort();
            assert_eq!(stacks2, vec![vec![0, 1, 41, 5], vec![0, 1, 384]]);
        }

        let mut cache_thrash = vec![0u8; 32 * 1024 * 1024];
        let mut thrash_checksum: u64 = 0;
        let mut cold_wall_ns: u128 = 0;

        for iter in 0..cold_iters {
            let salt = (iter as u8).wrapping_mul(17).wrapping_add(3);
            for i in (0..cache_thrash.len()).step_by(64) {
                cache_thrash[i] = cache_thrash[i].wrapping_add(salt);
                thrash_checksum = thrash_checksum.wrapping_add(cache_thrash[i] as u64);
            }

            let t0 = Instant::now();
            let gss2 = advance_stacks(&table, &gss1_base, 8);
            cold_wall_ns += t0.elapsed().as_nanos();

            let mut stacks2: Vec<Vec<u32>> = gss2.to_stacks().into_iter().map(|(s, _)| s).collect();
            stacks2.sort();
            assert_eq!(stacks2, vec![vec![0, 1, 41, 5], vec![0, 1, 384]]);
        }

        black_box(thrash_checksum);

        let warm_inv = 1.0 / warm_iters as f64;
        let cold_inv = 1.0 / cold_iters as f64;
        eprintln!(
            "[advance_manual_o1051_faithful][warm] avg_wall_us={:.3}",
            (warm_wall_ns as f64) * warm_inv / 1_000.0,
        );
        eprintln!(
            "[advance_manual_o1051_faithful][cold] avg_wall_us={:.3} cold_method=thrash_32MiB_stride64",
            (cold_wall_ns as f64) * cold_inv / 1_000.0,
        );
    }

    #[test]
    fn test_advance_manual_o1051_second_advance_component_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        let (table, gss0) = build_manual_o1051_faithful_table_and_stack();
        let gss1 = advance_stacks(&table, &gss0, 47);
        let iters = 50_000usize;

        let mut det_prelude_ns: u128 = 0;
        let mut split_det_probe_ns: u128 = 0;
        let mut shift_absorb_ns: u128 = 0;
        let mut reduce_sources_ns: u128 = 0;
        let mut reduce_branch_ns: u128 = 0;
        let mut reduce_absorb_ns: u128 = 0;
        let mut checksum: u64 = 0;

        for _ in 0..iters {
            let mut closure = gss1.clone();

            let t0 = Instant::now();
            let det_done = advance_deterministically(&table, &mut closure, 8);
            det_prelude_ns += t0.elapsed().as_nanos();
            assert!(!det_done);

            let state = closure.single_exclusive_top_value().unwrap();
            assert_eq!(state, 7);
            let action = table.action(state, 8).unwrap();

            let mut isolated = closure.isolate(Some(state));
            let reduce_base = isolated.clone();

            let t1 = Instant::now();
            let split_det_done = advance_deterministically(&table, &mut isolated, 8);
            split_det_probe_ns += t1.elapsed().as_nanos();
            assert!(!split_det_done);

            let shift_target = action.shift_target().unwrap();
            let shift_base = isolated.popn(1);
            let t2 = Instant::now();
            let mut shifted = ParserGSS::empty().absorb_push_same_acc(shift_target, &shift_base);
            shift_absorb_ns += t2.elapsed().as_nanos();

            let mut reduce_pair = None;
            action.for_each_reduce(|nt, len| {
                reduce_pair = Some((nt, len));
            });
            let (nt, len) = reduce_pair.unwrap();

            let t3 = Instant::now();
            let sources = reduce_sources_from_isolated(&reduce_base, len as usize);
            reduce_sources_ns += t3.elapsed().as_nanos();
            assert_eq!(sources.len(), 1);

            let (goto_from, base) = sources.into_iter().next().unwrap();
            let (target, is_replace) = table.goto_target(goto_from, nt).unwrap();

            let t4 = Instant::now();
            let (branch, det_ok) = advance_reduce_branch(&table, base, target, is_replace, 8);
            reduce_branch_ns += t4.elapsed().as_nanos();
            assert!(det_ok);

            let t5 = Instant::now();
            match branch.into_virtual_stack() {
                Ok(stack) => {
                    shifted = shifted.absorb_vstack_same_acc_owned(stack);
                }
                Err(branch) => {
                    merge_into(&mut shifted, branch);
                }
            }
            reduce_absorb_ns += t5.elapsed().as_nanos();

            let mut stacks2: Vec<Vec<u32>> = shifted.to_stacks().into_iter().map(|(s, _)| s).collect();
            stacks2.sort();
            assert_eq!(stacks2, vec![vec![0, 1, 41, 5], vec![0, 1, 384]]);
            checksum = checksum.wrapping_add(stacks2.len() as u64);
        }

        black_box(checksum);
        let inv = 1.0 / iters as f64;
        eprintln!(
            "[advance_manual_o1051_components][warm] det_prelude_us={:.3} split_det_probe_us={:.3} shift_absorb_us={:.3} reduce_sources_us={:.3} reduce_branch_us={:.3} reduce_absorb_us={:.3}",
            (det_prelude_ns as f64) * inv / 1_000.0,
            (split_det_probe_ns as f64) * inv / 1_000.0,
            (shift_absorb_ns as f64) * inv / 1_000.0,
            (reduce_sources_ns as f64) * inv / 1_000.0,
            (reduce_branch_ns as f64) * inv / 1_000.0,
            (reduce_absorb_ns as f64) * inv / 1_000.0,
        );
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
    fn test_manual_table_mre_abstract_split_path_regression_minimal() {
        // Regression test for unit-reduction inlining bug.
        // Grammar has two derivations for the "object" nonterminal N55:
        //   fused:  N55 → T(8) T(18)           — input [8, 18]
        //   split:  N55 → T(8) T(9) T(10)      — input [8, 9, 10]
        // T(9) also appears as the final token of the start rule, which is
        // the structural overlap that previously caused the inlining pass to
        // drop the split path from the table.
        let gdef = make_grammar(
            vec![
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(1), Symbol::Nonterminal(1)] },
                Rule { lhs: 55, rhs: vec![Symbol::Terminal(8), Symbol::Terminal(18)] },
                Rule { lhs: 55, rhs: vec![Symbol::Terminal(8), Symbol::Terminal(9), Symbol::Terminal(10)] },
                Rule { lhs: 58, rhs: vec![Symbol::Terminal(8)] },
                Rule { lhs: 59, rhs: vec![Symbol::Nonterminal(58), Symbol::Terminal(1), Symbol::Terminal(17), Symbol::Terminal(7), Symbol::Nonterminal(55)] },
                Rule { lhs: 60, rhs: vec![Symbol::Nonterminal(59), Symbol::Terminal(1), Symbol::Terminal(19), Symbol::Terminal(7), Symbol::Nonterminal(2)] },
                Rule { lhs: 27, rhs: vec![Symbol::Nonterminal(60), Symbol::Terminal(9)] },
            ],
            27,
            (0..20).map(|i| tdef(i, &format!("t{i}"))).collect(),
        );

        let grammar = AnalyzedGrammar::from_grammar_def(&gdef);
        let table_no_inline = GLRTable::build_with_unit_reduction_inlining(&grammar, false);
        let table_inline = GLRTable::build_with_unit_reduction_inlining(&grammar, true);
        let parser_no_inline = GLRParser::new(table_no_inline);
        let parser_inline = GLRParser::new(table_inline);

        // Both inputs are in the language of the grammar.
        let fused_input = vec![8, 1, 17, 7, 8, 18, 1, 19, 7, 1, 0, 9];
        let split_input = vec![8, 1, 17, 7, 8, 9, 10, 1, 19, 7, 1, 0, 9];

        assert!(accepts(&parser_no_inline, &fused_input), "no-inline: fused path must accept");
        assert!(accepts(&parser_no_inline, &split_input), "no-inline: split path must accept");
        assert!(accepts(&parser_inline, &fused_input), "inline: fused path must accept");
        assert!(accepts(&parser_inline, &split_input), "inline: split path must accept");
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
