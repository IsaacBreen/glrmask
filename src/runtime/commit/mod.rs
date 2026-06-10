pub(crate) mod profile;
mod template_advance;
pub(crate) mod tokenizer_scan;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::automata::lexer::tokenizer::{TokenizerExecResult, TokenizerMatch};
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    ParserGSS,
    apply_guarded_stack_shifts_fast,
    advance_stacks,
    advance_stacks_profiled,
    advance_stacks_owned,
    AdvanceProfile,
    stack_may_advance_on,
    stack_may_advance_on_any,
};
use crate::compiler::glr::table::{Action, GLRTable};
use crate::runtime::constraint::Constraint;
use crate::runtime::state::{CommitBuffers, ConstraintState};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use self::profile::{
    apply_advance_profile,
    fast_action_advance_profile,
    CommitProfile,
    PerAdvanceEntry,
};
use self::template_advance::{
    advance_stacks_template_dfa,
    advance_stacks_template_dfa_owned,
};
use self::tokenizer_scan::{execute_tokenizer_from_state_small, InitialCommitScan};

type ParserStatesByTokenizer = FxHashMap<u32, ParserGSS>;

const SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX: usize = 8;

#[derive(Clone, Copy)]
struct NormalizedMatch {
    terminal_id: u32,
    width: usize,
    ignored: bool,
}

const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 256;
static TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();
static VALIDATE_TEMPLATE_ADVANCE_ENABLED: OnceLock<bool> = OnceLock::new();

fn template_advance_enabled() -> bool {
    *TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_DISABLE_TEMPLATE_DFA_ADVANCE").is_none())
}

fn validate_template_advance_enabled() -> bool {
    *VALIDATE_TEMPLATE_ADVANCE_ENABLED
        .get_or_init(|| std::env::var_os("GLRMASK_VALIDATE_TEMPLATE_DFA_ADVANCE").is_some())
}

fn advance_parser_stacks(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if template_advance_enabled()
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks(&constraint.table, stack, terminal);
            assert_eq!(
                template_advanced,
                table_advanced,
                "template-DFA advance mismatch for terminal {terminal}"
            );
        }
        return template_advanced;
    }

    advance_stacks(&constraint.table, stack, terminal)
}

fn advance_parser_stacks_owned(
    constraint: &Constraint,
    stack: ParserGSS,
    terminal: u32,
) -> ParserGSS {
    if template_advance_enabled()
        && let Some(template_advanced) =
            advance_stacks_template_dfa_owned(constraint, stack.clone(), terminal)
    {
        if validate_template_advance_enabled() {
            let table_advanced = advance_stacks_owned(&constraint.table, stack, terminal);
            assert_eq!(
                template_advanced,
                table_advanced,
                "template-DFA advance mismatch for terminal {terminal}"
            );
        }
        return template_advanced;
    }

    advance_stacks_owned(&constraint.table, stack, terminal)
}

fn advance_parser_stacks_profiled(
    constraint: &Constraint,
    stack: &ParserGSS,
    terminal: u32,
) -> (ParserGSS, AdvanceProfile) {
    let template_start = std::time::Instant::now();
    if template_advance_enabled()
        && let Some(template_advanced) = advance_stacks_template_dfa(constraint, stack, terminal)
    {
        let template_elapsed = template_start.elapsed().as_nanos() as u64;
        if validate_template_advance_enabled() {
            let (table_advanced, table_profile) =
                advance_stacks_profiled(&constraint.table, stack, terminal);
            assert_eq!(
                template_advanced,
                table_advanced,
                "template-DFA advance mismatch for terminal {terminal}"
            );
            return (template_advanced, table_profile);
        }
        return (
            template_advanced,
            AdvanceProfile {
                total_ns: template_elapsed,
                fast_path_ns: template_elapsed,
                top_states: stack.peek_values().len() as u32,
                gss_depth: stack.max_depth(),
                vstack_len: stack
                    .try_virtual_stack()
                    .map_or(0, |vstack| vstack.len() as u32),
                ..AdvanceProfile::default()
            },
        );
    }

    advance_stacks_profiled(&constraint.table, stack, terminal)
}

/// Cache for `advance_stacks` results, keyed by (GSS pointer, terminal).
/// Stores the key GSS alongside the result to keep its Arc alive and prevent
/// address reuse (ABA problem) within a single `commit_bytes_impl` call.
type AdvanceResultCache = FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>;

fn state_has_nonempty_accumulators(state: &BTreeMap<u32, ParserGSS>) -> bool {
    state
        .values()
        .any(|gss| !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
}


fn parser_stacks_only(gss: &ParserGSS) -> Vec<Vec<u32>> {
    gss.to_stacks().into_iter().map(|(stack, _)| stack).collect()
}


fn token_bytes_for_id(constraint: &Constraint, token_id: u32) -> Option<&[u8]> {
    constraint
        .token_bytes_dense
        .get(token_id as usize)
        .and_then(|bytes| bytes.as_deref())
        .or_else(|| constraint.token_bytes.get(&token_id).map(Vec::as_slice))
}

fn commit_mask_assert_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if cfg!(debug_assertions) {
            return true;
        }
        std::env::var("GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE")
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

fn profile_allow_fast_paths() -> bool {
    std::env::var("GLRMASK_PROFILE_ALLOW_FAST_PATHS")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn token_in_mask(mask: &[u32], token_id: u32) -> bool {
    let word_idx = token_id as usize / 32;
    let bit_idx = token_id as usize % 32;
    word_idx < mask.len() && ((mask[word_idx] >> bit_idx) & 1) != 0
}

fn snapshot_mask_membership(state: &ConstraintState<'_>, token_id: u32) -> Option<bool> {
    if !commit_mask_assert_enabled() {
        return None;
    }
    let mut mask = vec![0u32; state.constraint.mask_len()];
    state.fill_mask(&mut mask);
    Some(token_in_mask(&mask, token_id))
}

fn format_token_bytes(token_bytes: &[u8]) -> String {
    let mut escaped = String::new();
    for byte in token_bytes {
        for ch in std::ascii::escape_default(*byte) {
            escaped.push(ch as char);
        }
    }
    format!("b\"{}\"", escaped)
}

pub(crate) fn commit_bytes_accepts_for_mask(
    constraint: &Constraint,
    state: &BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> bool {
    if bytes.is_empty() {
        return true;
    }
    let mut cloned_state = state.clone();
    let mut buffers = CommitBuffers::default();
    commit_bytes_impl(constraint, &mut cloned_state, bytes, &mut buffers).is_ok()
}

fn assert_mask_commit_equivalence(
    token_id: u32,
    token_bytes: &[u8],
    was_in_mask: Option<bool>,
    commit_succeeded: bool,
) {
    let Some(was_in_mask) = was_in_mask else {
        return;
    };
    assert!(
        commit_succeeded == was_in_mask,
        "commit/mask mismatch for token_id {} bytes {}: token_in_mask={} commit_succeeded={}",
        token_id,
        format_token_bytes(token_bytes),
        was_in_mask,
        commit_succeeded,
    );
}

#[inline]
fn end_state_may_advance(constraint: &Constraint, gss: &ParserGSS, end_state: u32) -> bool {
    end_state == constraint.tokenizer.initial_state()
        || stack_may_advance_on_any(
            &constraint.table,
            gss,
            constraint.tokenizer.possible_future_terminals(end_state),
        )
}

enum ActionableTerminals {
    SingleState(u32),
    ManyStates(SmallVec<[u32; 8]>),
}

impl ActionableTerminals {
    fn from_gss(_constraint: &Constraint, gss: &ParserGSS) -> Option<Self> {
        if let Some(state_id) = gss.single_top_value() {
            return Some(Self::SingleState(state_id));
        }

        let states = gss.peek_values();
        if states.is_empty() {
            None
        } else {
            Some(Self::ManyStates(states))
        }
    }

    fn contains(&self, constraint: &Constraint, terminal: u32) -> bool {
        match self {
            Self::SingleState(state_id) => constraint.table.advance_row_allows(*state_id, terminal),
            Self::ManyStates(states) => states
                .iter()
                .any(|state_id| constraint.table.advance_row_allows(*state_id, terminal)),
        }
    }
}

impl InitialCommitScan {
    fn collect(
        constraint: &Constraint,
        state: &BTreeMap<u32, ParserGSS>,
        bytes: &[u8],
    ) -> Self {
        let ignore_terminal = constraint.ignore_terminal;
        let mut exec_results = FxHashMap::default();
        let mut remapped_tokenizer_states = FxHashMap::default();
        let mut accepted_terminals = FxHashMap::<u32, FxHashSet<u32>>::default();

        for (&tokenizer_state, parser_gss) in state {
            let actionable_terminals = ActionableTerminals::from_gss(constraint, parser_gss);
            let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);

            if let Some(end_state) = exec_result.end_state {
                remapped_tokenizer_states.insert(tokenizer_state, end_state);
            }

            for matched in &exec_result.matches {
                if is_ignored_terminal(ignore_terminal, matched.id)
                    || !is_actionable_terminal(
                        actionable_terminals.as_ref(),
                        constraint,
                        matched.id,
                    )
                {
                    continue;
                }

                accepted_terminals
                    .entry(tokenizer_state)
                    .or_default()
                    .insert(matched.id);
            }

            exec_results.insert(tokenizer_state, exec_result);
        }

        Self {
            exec_results,
            remapped_tokenizer_states,
            accepted_terminals,
        }
    }

    fn take_exec_result(&mut self, tokenizer_state: u32) -> Option<TokenizerExecResult> {
        self.exec_results.remove(&tokenizer_state)
    }
}

fn is_ignored_terminal(ignore_terminal: Option<u32>, terminal: u32) -> bool {
    Some(terminal) == ignore_terminal
}

fn is_actionable_terminal(
    actionable_terminals: Option<&ActionableTerminals>,
    constraint: &Constraint,
    terminal: u32,
) -> bool {
    !actionable_terminals
        .is_some_and(|actionable| !actionable.contains(constraint, terminal))
}

fn collect_unique_actionable_matches(
    constraint: &Constraint,
    actionable_terminals: Option<&ActionableTerminals>,
    ignore_terminal: Option<u32>,
    matches: &[TokenizerMatch],
    reusable_seen_matches: Option<&mut FxHashSet<(usize, u32)>>,
) -> SmallVec<[NormalizedMatch; 8]> {
    let mut normalized = SmallVec::<[NormalizedMatch; 8]>::new();

    if matches.len() <= SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX {
        'matches: for matched in matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            for existing in &normalized {
                if existing.width == matched.width && existing.terminal_id == matched.id {
                    continue 'matches;
                }
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    if let Some(seen_matches) = reusable_seen_matches {
        seen_matches.clear();
        for matched in matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
                continue;
            }
            if !seen_matches.insert((matched.width, matched.id)) {
                continue;
            }
            normalized.push(NormalizedMatch {
                terminal_id: matched.id,
                width: matched.width,
                ignored,
            });
        }
        return normalized;
    }

    let mut seen_matches = FxHashSet::default();
    for matched in matches {
        let ignored = is_ignored_terminal(ignore_terminal, matched.id);
        if !ignored && !is_actionable_terminal(actionable_terminals, constraint, matched.id) {
            continue;
        }
        if !seen_matches.insert((matched.width, matched.id)) {
            continue;
        }
        normalized.push(NormalizedMatch {
            terminal_id: matched.id,
            width: matched.width,
            ignored,
        });
    }
    normalized
}

fn prune_initial_states(
    state: &mut BTreeMap<u32, ParserGSS>,
    accepted_terminals: &FxHashMap<u32, FxHashSet<u32>>,
    remapped_tokenizer_states: &FxHashMap<u32, u32>,
) {
    if accepted_terminals.is_empty()
        && state
            .values()
            .all(|parser_state| parser_state.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
    {
        return;
    }

    for parser_state in state.values_mut() {
        *parser_state = parser_state.apply_and_prune_no_promote(
            |terminals_disallowed: &TerminalsDisallowed| {
                for (tokenizer_state, matched_terminals) in accepted_terminals {
                    if let Some(disallowed) = terminals_disallowed.get(tokenizer_state) {
                        if !matched_terminals.is_empty()
                            && matched_terminals
                                .iter()
                                .all(|terminal| disallowed.contains(terminal))
                        {
                            return None;
                        }
                    }
                }

                let mut remapped = BTreeMap::new();
                for (old_state, new_state) in remapped_tokenizer_states {
                    if let Some(disallowed) = terminals_disallowed.get(old_state) {
                        remapped
                            .entry(*new_state)
                            .or_insert_with(BTreeSet::new)
                            .extend(disallowed.iter().copied());
                    }
                }
                Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
            },
        );
    }
}

fn prune_single_initial_state_for_exec(
    constraint: &Constraint,
    gss: ParserGSS,
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
) -> ParserGSS {
    let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss);
    let mut accepted_terminals: SmallVec<[u32; 4]> = SmallVec::new();
    for matched in &exec_result.matches {
        if is_ignored_terminal(constraint.ignore_terminal, matched.id) {
            continue;
        }
        if is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id) {
            accepted_terminals.push(matched.id);
        }
    }

    if accepted_terminals.is_empty() && exec_result.end_state.is_none() {
        return gss;
    }

    if accepted_terminals.is_empty()
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
    {
        return gss;
    }

    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        if terminals_disallowed.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
            if !accepted_terminals.is_empty()
                && accepted_terminals
                    .iter()
                    .all(|terminal| disallowed.contains(terminal))
            {
                return None;
            }
        }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = exec_result.end_state {
            if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
                remapped
                    .entry(end_state)
                    .or_insert_with(BTreeSet::new)
                    .extend(disallowed.iter().copied());
            }
        }
        Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
    })
}

fn prune_single_initial_state_for_terminal(
    gss: ParserGSS,
    tokenizer_state: u32,
    terminal: u32,
    end_state: Option<u32>,
) -> ParserGSS {
    if end_state.is_none()
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| {
            td.get(&tokenizer_state)
                .is_none_or(|disallowed| !disallowed.contains(&terminal))
        })
    {
        return gss.apply(|_: &TerminalsDisallowed| TerminalsDisallowed::new());
    }

    gss.apply_and_prune_no_promote(|terminals_disallowed: &TerminalsDisallowed| {
        if terminals_disallowed.is_empty() {
            return Some(TerminalsDisallowed::new());
        }
        if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
            if disallowed.contains(&terminal) {
                return None;
            }
        }

        let mut remapped = BTreeMap::new();
        if let Some(end_state) = end_state {
            if let Some(disallowed) = terminals_disallowed.get(&tokenizer_state) {
                remapped
                    .entry(end_state)
                    .or_insert_with(BTreeSet::new)
                    .extend(disallowed.iter().copied());
            }
        }
        Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
    })
}

fn merge_parser_state(
    states: &mut ParserStatesByTokenizer,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    states
        .entry(tokenizer_state)
        .and_modify(|existing| *existing = existing.merge(&gss))
        .or_insert(gss);
}

fn queue_parser_state(
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    if new_offset == total_len {
        merge_parser_state(pending_state, tokenizer_state, gss);
    } else {
        merge_parser_state(&mut processing_queue[new_offset], tokenizer_state, gss);
    }
}

fn finalize_pending_state(mut pending_state: ParserStatesByTokenizer) -> BTreeMap<u32, ParserGSS> {
    match pending_state.len() {
        0 => BTreeMap::new(),
        1 => {
            let (tokenizer_state, parser_state) = pending_state.drain().next().unwrap();
            let fused = parser_state.fuse(Some(1));
            if fused.is_empty() {
                BTreeMap::new()
            } else {
                let mut new_state = BTreeMap::new();
                new_state.insert(tokenizer_state, fused);
                new_state
            }
        }
        _ => {
            let mut new_state: BTreeMap<u32, ParserGSS> = pending_state.into_iter().collect();
            for parser_state in new_state.values_mut() {
                *parser_state = parser_state.fuse(Some(1));
            }
            new_state.retain(|_, parser_state| !parser_state.is_empty());
            new_state
        }
    }
}

fn apply_future_terminal_disallow(
    constraint: &Constraint,
    exec_result: &TokenizerExecResult,
    terminal: u32,
    gss: ParserGSS,
) -> ParserGSS {
    if gss.is_empty() {
        return gss;
    }

    let Some(end_state) = exec_result.end_state else {
        return gss;
    };
    if !constraint
        .tokenizer
        .dfa
        .possible_future_group_ids(end_state)
        .contains(terminal as usize)
    {
        return gss;
    }

    gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
        terminals_disallowed.with_insert(end_state, terminal)
    })
}

#[inline]
fn apply_single_top_action_fast(
    table: &GLRTable,
    gss: &ParserGSS,
    state: u32,
    terminal: u32,
    action: &Action,
) -> Option<ParserGSS> {
    match action {
        Action::Shift(target, is_replace) => {
            if let Some(mut stack) = gss.try_virtual_stack() {
                if *is_replace && stack.pop(1) != 0 {
                    return Some(gss.popn(1).push(*target));
                }
                stack.push(*target);
                return Some(stack.into_gss());
            } else {
                Some(if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                })
            }
        }
        Action::StackShifts(shifts) => {
            if let [shift] = shifts.as_slice() {
                let mut branch = gss.try_virtual_stack()?;
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                return Some(branch.into_gss());
            }
            if let Some(stack) = gss.try_virtual_stack()
                && let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = stack.into_gss_after_popping_and_pushing_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }
            if let Some(shifted) = gss.apply_stack_effects_to_single_concrete_path(
                shifts
                    .iter()
                    .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
            ) {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && shift.pushes.len() == 1)
                && let Some(shifted) = gss.apply_shared_pop_push_single_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| &shift.pushes[0]),
                )
            {
                return Some(shifted);
            }
            if let Some(first) = shifts.first()
                && !first.pushes.is_empty()
                && shifts
                    .iter()
                    .all(|shift| shift.pop == first.pop && !shift.pushes.is_empty())
                && let Some(shifted) = gss.apply_shared_pop_push_branches(
                    first.pop as usize,
                    shifts.iter().map(|shift| shift.pushes.as_slice()),
                )
            {
                return Some(shifted);
            }

            let stack = gss.try_virtual_stack()?;
            let mut shifted = ParserGSS::empty();
            for shift in shifts {
                let mut branch = stack.clone();
                if branch.pop(shift.pop as usize) != 0 {
                    return None;
                }
                for &target in &shift.pushes {
                    branch.push(target);
                }
                let branch = branch.into_gss();
                shifted = if shifted.is_empty() {
                    branch
                } else {
                    shifted.merge(&branch)
                };
            }
            Some(shifted)
        }
        Action::GuardedStackShifts(shifts) => {
            apply_guarded_stack_shifts_fast(gss, shifts, table.guarded_shift_index(state, terminal))
        }
        Action::Reduce(..) => apply_single_path_reduce_chain_fast(table, gss, terminal),
        _ => None,
    }
}

fn apply_single_path_reduce_chain_fast(
    table: &GLRTable,
    gss: &ParserGSS,
    terminal: u32,
) -> Option<ParserGSS> {
    let mut top_first = SmallVec::<[u32; 16]>::new();
    let acc = gss.single_path_top_first_and_acc(&mut top_first)?;
    if top_first.len() > SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH {
        return None;
    }

    let mut stack = top_first.into_vec();
    stack.reverse();

    loop {
        let state = *stack.last()?;
        match table.action(state, terminal)? {
            Action::Reduce(nt, len) => {
                let rhs_len = *len as usize;
                if rhs_len >= stack.len() {
                    return None;
                }
                stack.truncate(stack.len() - rhs_len);
                let goto_from = *stack.last()?;
                let (target, is_replace) = table.goto_target(goto_from, *nt)?;
                if is_replace {
                    *stack.last_mut()? = target;
                } else {
                    stack.push(target);
                }
            }
            Action::Shift(target, is_replace) => {
                if *is_replace {
                    *stack.last_mut()? = *target;
                } else {
                    stack.push(*target);
                }
                return Some(ParserGSS::from_single_stack(stack, acc));
            }
            Action::StackShifts(shifts) => {
                return ParserGSS::from_single_stack(stack, acc)
                    .apply_stack_effects_to_single_concrete_path(
                        shifts
                            .iter()
                            .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                        SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                    );
            }
            Action::Split {
                shift,
                reduces,
                accept: false,
            } => {
                let mut out: Vec<(Vec<u32>, TerminalsDisallowed)> = Vec::new();

                if let Some((target, is_replace)) = shift {
                    let mut branch = stack.clone();
                    if *is_replace {
                        *branch.last_mut()? = *target;
                    } else {
                        branch.push(*target);
                    }
                    out.push((branch, acc.clone()));
                }

                for &(nt, len) in reduces {
                    let mut branch = stack.clone();
                    let rhs_len = len as usize;
                    if rhs_len >= branch.len() {
                        return None;
                    }
                    branch.truncate(branch.len() - rhs_len);
                    let goto_from = *branch.last()?;
                    let (target, is_replace) = table.goto_target(goto_from, nt)?;
                    if is_replace {
                        *branch.last_mut()? = target;
                    } else {
                        branch.push(target);
                    }

                    let follow_state = *branch.last()?;
                    match table.action(follow_state, terminal)? {
                        Action::Shift(target, is_replace) => {
                            if *is_replace {
                                *branch.last_mut()? = *target;
                            } else {
                                branch.push(*target);
                            }
                            out.push((branch, acc.clone()));
                        }
                        Action::StackShifts(shifts) => {
                            let shifted = ParserGSS::from_single_stack(branch, acc.clone())
                                .apply_stack_effects_to_single_concrete_path(
                                    shifts
                                        .iter()
                                        .map(|shift| (shift.pop as usize, shift.pushes.as_slice())),
                                    SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH,
                                )?;
                            out.extend(shifted.to_stacks());
                        }
                        _ => return None,
                    }
                }

                return (!out.is_empty()).then(|| ParserGSS::from_stacks(&out));
            }
            _ => return None,
        }
    }
}

fn advance_terminal_match(
    constraint: &Constraint,
    gss_at_offset: &ParserGSS,
    terminal: u32,
    exec_result: &TokenizerExecResult,
    advance_result_cache: &mut AdvanceResultCache,
    terminal_result_cache: &mut FxHashMap<u32, ParserGSS>,
) -> Option<ParserGSS> {
    if let Some(cached) = terminal_result_cache.get(&terminal) {
        return (!cached.is_empty()).then(|| cached.clone());
    }

    let advance_cache_key = (gss_at_offset.ptr_key(), terminal);
    let advanced = if let Some((_, cached)) = advance_result_cache.get(&advance_cache_key) {
        cached.clone()
    } else {
        let advanced = advance_parser_stacks(constraint, gss_at_offset, terminal);
        advance_result_cache.insert(advance_cache_key, (gss_at_offset.clone(), advanced.clone()));
        advanced
    };

    let advanced = apply_future_terminal_disallow(constraint, exec_result, terminal, advanced);
    terminal_result_cache.insert(terminal, advanced.clone());
    (!advanced.is_empty()).then_some(advanced)
}

/// Fast path for the common case: exactly 1 tokenizer state, the tokenizer
/// produces exactly 1 non-ignored terminal match that consumes all bytes,
/// and no pending end-state needs to be queued. This avoids:
/// - FxHashMap allocations (InitialCommitScan, seen_matches, caches)
/// - Processing queue allocation
/// - Prune iteration (when terminals_disallowed is empty)
///
/// Returns `Some(Ok(()))` on success, `Some(Err(...))` on rejection,
/// or `None` to fall through to the general path.
///
/// `exec_result` is the pre-computed tokenizer output for the single state.
fn commit_bytes_fast_path(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
) -> Option<Result<(), String>> {
    let gss = state.values().next().unwrap();
    let ignore_terminal = constraint.ignore_terminal;

    // Find exactly 1 non-ignored, actionable terminal match consuming all bytes
    let mut sole_terminal: Option<u32> = None;
    for matched in &exec_result.matches {
        if matched.width != bytes.len() {
            return None;
        }
        if is_ignored_terminal(ignore_terminal, matched.id) {
            return None;
        }
        if !stack_may_advance_on(&constraint.table, gss, matched.id) {
            continue;
        }
        if sole_terminal.is_some() {
            return None;
        }
        sole_terminal = Some(matched.id);
    }
    let terminal = sole_terminal?;

    let no_end_state = exec_result.end_state.is_none();
    let accs_empty = gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());
    let all_accs_empty = no_end_state && accs_empty;

    // Ultra-fast path: single Interface, empty accs, no end_state, pure shift.
    // Inlines the entire advance + prune + fuse to avoid all function call overhead.
    if all_accs_empty && !template_advance_enabled() {
        if let Some(top_state) = gss.single_exclusive_top_value() {
            if let Some(action) = constraint.table.action(top_state, terminal) {
                if let Some(shifted) =
                    apply_single_top_action_fast(&constraint.table, gss, top_state, terminal, action)
                {
                    state.clear();
                    state.insert(constraint.tokenizer.initial_state(), shifted);
                    return Some(Ok(()));
                }
            }
        }
    }

    // Take ownership of the GSS for the standard fast path.
    // This allows advance_stacks_owned to avoid cloning the inner Arc.
    let (_, gss_owned) = state.pop_first().unwrap();

    // Standard fast path: skip prune when accumulators are empty.
    let pruned_gss = if accs_empty {
        gss_owned
    } else {
        let pruned = gss_owned.apply_and_prune_no_promote(|td: &TerminalsDisallowed| {
            if td.is_empty() {
                return Some(TerminalsDisallowed::new());
            }
            if let Some(disallowed) = td.get(&tokenizer_state) {
                if disallowed.contains(&terminal) {
                    return None;
                }
            }
            let mut remapped = BTreeMap::new();
            if let Some(end_state) = exec_result.end_state {
                if let Some(d) = td.get(&tokenizer_state) {
                    remapped
                        .entry(end_state)
                        .or_insert_with(BTreeSet::new)
                        .extend(d.iter().copied());
                }
            }
            Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
        });

        if pruned.is_empty() {
            return Some(Err(
                "commit rejected: no valid parser states remain".to_string(),
            ));
        }
        pruned
    };

    let end_state_to_keep = exec_result
        .end_state
        .filter(|&end_state| end_state_may_advance(constraint, &pruned_gss, end_state));
    let end_state_gss = end_state_to_keep.map(|_| pruned_gss.clone());

    // The terminal and tokenizer end-state continuations are independent.
    // Preserve either branch if it produces viable parser state.
    let advanced = if !template_advance_enabled()
        && let Some(top_state) = pruned_gss.single_exclusive_top_value()
        && let Some(action) = constraint.table.action(top_state, terminal)
        && let Some(advanced) = apply_single_top_action_fast(
            &constraint.table,
            &pruned_gss,
            top_state,
            terminal,
            action,
        )
    {
        advanced
    } else {
        advance_parser_stacks_owned(constraint, pruned_gss, terminal)
    };
    let mut produced_state = false;
    if !advanced.is_empty() {
        let advanced =
            apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
        if !advanced.is_empty() {
            let fused = advanced.fuse(Some(1));
            if !fused.is_empty() {
                state.insert(constraint.tokenizer.initial_state(), fused);
                produced_state = true;
            }
        }
    }

    if let (Some(end_state), Some(end_gss)) = (end_state_to_keep, end_state_gss) {
        let fused = end_gss.fuse(Some(1));
        if !fused.is_empty() {
            state
                .entry(end_state)
                .and_modify(|existing| *existing = existing.merge(&fused))
                .or_insert(fused);
            produced_state = true;
        }
    }

    if !produced_state {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    Some(Ok(()))
}

fn commit_bytes_full_width_fast_path(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Option<Result<(), String>> {
    if state.len() > 2 {
        return None;
    }
    if state.len() > 1 && bytes.len() > 4 && state_has_nonempty_accumulators(state) {
        return None;
    }

    let mut output = ParserStatesByTokenizer::default();
    for (&tokenizer_state, gss) in state.iter() {
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        let actionable_terminals = ActionableTerminals::from_gss(constraint, gss);
        let mut terminal = None;

        for matched in &exec_result.matches {
            if matched.width != bytes.len()
                || is_ignored_terminal(constraint.ignore_terminal, matched.id)
            {
                return None;
            }
            if !is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id) {
                continue;
            }
            if terminal.is_some_and(|existing| existing != matched.id) {
                return None;
            }
            terminal = Some(matched.id);
        }

        let pruned_gss = if gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
            gss.clone()
        } else {
            let pruned = prune_single_initial_state_for_exec(
                constraint,
                gss.clone(),
                tokenizer_state,
                &exec_result,
            );
            if pruned.is_empty() {
                continue;
            }
            pruned
        };

        if let Some(terminal) = terminal {
            let advanced = if !template_advance_enabled()
                && let Some(top_state) = pruned_gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(
                        &constraint.table,
                        &pruned_gss,
                        top_state,
                        terminal,
                        action,
                    )
            {
                advanced
            } else {
                if !stack_may_advance_on(&constraint.table, &pruned_gss, terminal) {
                    return None;
                }
                advance_parser_stacks(constraint, &pruned_gss, terminal)
            };
            if advanced.is_empty() {
                continue;
            }
            let advanced = apply_future_terminal_disallow(
                constraint,
                &exec_result,
                terminal,
                advanced,
            );
            if !advanced.is_empty() {
                merge_parser_state(
                    &mut output,
                    constraint.tokenizer.initial_state(),
                    advanced,
                );
            }
        }

        if let Some(end_state) = exec_result.end_state {
            if end_state_may_advance(constraint, &pruned_gss, end_state) {
                merge_parser_state(&mut output, end_state, pruned_gss);
            }
        }
    }

    let new_state = finalize_pending_state(output);
    if new_state.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    *state = new_state;
    Some(Ok(()))
}

fn merge_small_parser_state(
    states: &mut SmallVec<[(u32, ParserGSS); 4]>,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    for (existing_state, existing_gss) in states.iter_mut() {
        if *existing_state == tokenizer_state {
            *existing_gss = existing_gss.merge(&gss);
            return;
        }
    }
    states.push((tokenizer_state, gss));
}

fn commit_bytes_small_queue_fast_path(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Option<Result<(), String>> {
      if bytes.len() > 8 || state.len() > 2 {
          return None;
      }

      let mut processing_queue: Vec<SmallVec<[(u32, ParserGSS); 4]>> =
          (0..=bytes.len()).map(|_| SmallVec::new()).collect();
    for (&tokenizer_state, gss) in state.iter() {
        processing_queue[0].push((tokenizer_state, gss.clone()));
    }

    let mut pending_state = ParserStatesByTokenizer::default();
      let mut offset = 0usize;
      while offset <= bytes.len() {
          if processing_queue[offset].is_empty() {
              offset += 1;
              continue;
          }

          let states_to_process = std::mem::take(&mut processing_queue[offset]);
          let initial_tokenizer_state = constraint.tokenizer.initial_state();
          let mut initial_gss_at_offset: Option<ParserGSS> = None;
          for (tokenizer_state, gss) in &states_to_process {
              if *tokenizer_state != initial_tokenizer_state {
                  continue;
              }
              initial_gss_at_offset = Some(match initial_gss_at_offset.take() {
                  Some(existing) => existing.merge(gss),
                  None => gss.clone(),
              });
          }
          for (tokenizer_state, mut gss_at_offset) in states_to_process {
              let exec_result =
                  execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state);

            if offset == 0
                && !gss_at_offset.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            {
                gss_at_offset = prune_single_initial_state_for_exec(
                    constraint,
                    gss_at_offset,
                    tokenizer_state,
                    &exec_result,
                );
                if gss_at_offset.is_empty() {
                    continue;
                }
            }

            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                actionable_terminals.as_ref(),
                constraint.ignore_terminal,
                &exec_result.matches,
                None,
            );
            let mut emitted_terminal_outputs = SmallVec::<[(usize, ParserGSS); 4]>::new();

            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if new_offset > bytes.len() {
                    return None;
                }

                  if matched.ignored {
                      if new_offset == bytes.len() {
                          merge_parser_state(
                              &mut pending_state,
                              initial_tokenizer_state,
                              gss_at_offset.clone(),
                          );
                      } else {
                          merge_small_parser_state(
                              &mut processing_queue[new_offset],
                              initial_tokenizer_state,
                              gss_at_offset.clone(),
                          );
                      }
                      continue;
                  }

                  let advanced = if tokenizer_state != initial_tokenizer_state
                      && let Some(existing_initial_gss) = initial_gss_at_offset.as_ref()
                      && existing_initial_gss.all_accs_satisfy(|td: &TerminalsDisallowed| {
                          td.get(&tokenizer_state)
                              .is_some_and(|disallowed| disallowed.contains(&matched.terminal_id))
                      }) {
                      existing_initial_gss.clone()
                  } else if !template_advance_enabled()
                      && let Some(top_state) = gss_at_offset.single_exclusive_top_value()
                      && let Some(action) = constraint.table.action(top_state, matched.terminal_id)
                      && let Some(advanced) = apply_single_top_action_fast(
                        &constraint.table,
                        &gss_at_offset,
                        top_state,
                        matched.terminal_id,
                        action,
                    )
                {
                    advanced
                } else {
                    if !stack_may_advance_on(
                        &constraint.table,
                        &gss_at_offset,
                        matched.terminal_id,
                    ) {
                        continue;
                    }
                    advance_parser_stacks(constraint, &gss_at_offset, matched.terminal_id)
                };
                let advanced = apply_future_terminal_disallow(
                    constraint,
                    &exec_result,
                    matched.terminal_id,
                    advanced,
                );
                if advanced.is_empty() {
                    continue;
                }
                if emitted_terminal_outputs
                    .iter()
                    .any(|(emitted_offset, emitted_gss)| {
                        *emitted_offset == new_offset && emitted_gss == &advanced
                    })
                {
                    continue;
                }
                emitted_terminal_outputs.push((new_offset, advanced.clone()));
                  if new_offset == bytes.len() {
                      merge_parser_state(
                          &mut pending_state,
                          initial_tokenizer_state,
                          advanced,
                      );
                  } else {
                      merge_small_parser_state(
                          &mut processing_queue[new_offset],
                          initial_tokenizer_state,
                          advanced,
                      );
                  }
            }

            if let Some(end_state) = exec_result.end_state {
                if end_state_may_advance(constraint, &gss_at_offset, end_state) {
                    merge_parser_state(&mut pending_state, end_state, gss_at_offset);
                }
            }
        }
        offset += 1;
    }

    let new_state = finalize_pending_state(pending_state);
    if new_state.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    *state = new_state;
    Some(Ok(()))
}

enum LinearFastPathResult {
    Complete(Result<ParserGSS, String>),
    Continue { gss: ParserGSS, offset: usize },
    Restart,
}

struct DirectLinearStep {
    width: usize,
    terminal: u32,
    ignored: bool,
    end_state: Option<u32>,
}

fn choose_direct_linear_step(
    constraint: &Constraint,
    gss: &ParserGSS,
    bytes: &[u8],
    start_state: u32,
    carried_top_state: Option<u32>,
) -> Option<DirectLinearStep> {
    let ignore_terminal = constraint.ignore_terminal;
    let mut tokenizer_state = start_state;
    let mut chosen: Option<(usize, u32, bool)> = None;
    let mut consumed_all = true;
    let mut actionable_terminals = carried_top_state.map(ActionableTerminals::SingleState);

    for (index, &byte) in bytes.iter().enumerate() {
        let next_state = constraint
            .tokenizer_fast_transitions
            .get(tokenizer_state as usize)
            .map_or(u32::MAX, |transitions| transitions[byte as usize]);
        if next_state == u32::MAX {
            consumed_all = false;
            break;
        };
        tokenizer_state = next_state;
        let width = index + 1;
        let mut chosen_at_width = false;

        for terminal in constraint.tokenizer.matched_terminals_iter(tokenizer_state) {
            let ignored = is_ignored_terminal(ignore_terminal, terminal);
            if !ignored {
                if actionable_terminals.is_none() {
                    actionable_terminals = ActionableTerminals::from_gss(constraint, gss);
                }
                if !is_actionable_terminal(actionable_terminals.as_ref(), constraint, terminal) {
                    continue;
                }
            }

            let candidate = (width, terminal, ignored);
            chosen_at_width = true;
            if let Some((_, existing_terminal, _)) = chosen {
                if existing_terminal == terminal {
                    chosen = Some(candidate);
                } else {
                    return None;
                }
            } else {
                chosen = Some(candidate);
            }
        }

        if chosen_at_width && chosen.is_some_and(|(_, _, ignored)| ignored) {
            return Some(DirectLinearStep {
                width,
                terminal: chosen.unwrap().1,
                ignored: true,
                end_state: None,
            });
        }

        if chosen_at_width
            && chosen.is_some_and(|(_, _, ignored)| !ignored)
            && index + 1 < bytes.len()
        {
            let next_byte = bytes[index + 1];
            let next_state = constraint
                .tokenizer_fast_transitions
                .get(tokenizer_state as usize)
                .map_or(u32::MAX, |transitions| transitions[next_byte as usize]);
            if next_state == u32::MAX {
                let (_, terminal, _) = chosen.unwrap();
                return Some(DirectLinearStep {
                    width,
                    terminal,
                    ignored: false,
                    end_state: None,
                });
            }
        }
    }

    let (width, terminal, ignored) = chosen?;
    let end_state = consumed_all.then_some(tokenizer_state);

    Some(DirectLinearStep {
        width,
        terminal,
        ignored,
        end_state,
    })
}

fn commit_bytes_direct_linear_fast_path(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    start_tokenizer_state: u32,
    mut profile: Option<&mut CommitProfile>,
) -> Option<LinearFastPathResult> {
    let mut gss = start_gss;
    let mut carried_stack = gss.try_virtual_stack();
    let mut offset = 0usize;
    let mut tokenizer_state = start_tokenizer_state;

    while offset < bytes.len() {
        let choose_start = profile.as_ref().map(|_| std::time::Instant::now());
        let carried_top_state = carried_stack.as_ref().and_then(|stack| stack.top().copied());
        let Some(step) = choose_direct_linear_step(
            constraint,
            &gss,
            &bytes[offset..],
            tokenizer_state,
            carried_top_state,
        ) else {
            if let Some(stack) = carried_stack.take() {
                let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                gss = stack.into_gss();
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                    profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                }
            }
            if offset > 0 && profile.is_none() {
                return Some(LinearFastPathResult::Continue { gss, offset });
            }
            return None;
        };
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), choose_start) {
            profile.linear_fast_path_match_scan_ns += start.elapsed().as_nanos() as u64;
            profile.linear_fast_path_steps += 1;
        }

        let keep_carried = if let Some(end_state) = step.end_state
            && let Some(stack) = carried_stack.as_ref()
        {
            let carried_gate_start = profile.as_ref().map(|_| std::time::Instant::now());
            let keep_carried = stack.top().copied().is_some_and(|top_state| {
                end_state != constraint.tokenizer.initial_state()
                    && !constraint.table.advance_row_intersects(
                        top_state,
                        constraint.tokenizer.possible_future_terminals(end_state),
                    )
                    && !constraint
                        .tokenizer
                        .dfa
                        .possible_future_group_ids(end_state)
                        .contains(step.terminal as usize)
            });
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), carried_gate_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.linear_fast_path_carried_gate_ns += elapsed;
                profile.linear_fast_path_end_state_check_ns += elapsed;
            }
            keep_carried
        } else {
            false
        };
        if step.end_state.is_some() {
            if !keep_carried {
                if let Some(stack) = carried_stack.take() {
                    let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                    gss = stack.into_gss();
                    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                        profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                    }
                }
            }
        }

        if let Some(end_state) = step.end_state {
            let carried_gate_start = profile.as_ref().map(|_| std::time::Instant::now());
            let should_restart = end_state_may_advance(constraint, &gss, end_state);
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), carried_gate_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.linear_fast_path_carried_gate_ns += elapsed;
                profile.linear_fast_path_end_state_check_ns += elapsed;
            }
            if should_restart {
                if let Some(stack) = carried_stack.take() {
                    let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                    gss = stack.into_gss();
                    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                        profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                    }
                }
                if offset > 0 && profile.is_none() {
                    return Some(LinearFastPathResult::Continue { gss, offset });
                }
                return None;
            }
        }

        if !step.ignored {
            if offset == 0 {
                if !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
                    if let Some(stack) = carried_stack.take() {
                        let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                        gss = stack.into_gss();
                        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                            profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                        }
                    }
                    let prune_start = profile.as_ref().map(|_| std::time::Instant::now());
                    gss = prune_single_initial_state_for_terminal(
                        gss,
                        tokenizer_state,
                        step.terminal,
                        step.end_state,
                    );
                    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), prune_start) {
                        profile.prune_ns += start.elapsed().as_nanos() as u64;
                    }
                    if gss.is_empty() {
                        return Some(LinearFastPathResult::Complete(Err(
                            "commit rejected: no valid parser states remain".to_string(),
                        )));
                    }
                    carried_stack = gss.try_virtual_stack();
                }
            }

            let mut shifted_carried_stack = false;
            let mut carried_apply_elapsed_ns = 0u64;
            let action_lookup_start = profile.as_ref().map(|_| std::time::Instant::now());
            let carried_action = if let Some(stack) = carried_stack.as_ref()
                && let Some(top_state) = stack.top().copied()
                && step.end_state.is_none_or(|end_state| {
                    end_state != constraint.tokenizer.initial_state()
                        && !constraint.table.advance_row_intersects(
                            top_state,
                            constraint.tokenizer.possible_future_terminals(end_state),
                        )
                        && !constraint
                            .tokenizer
                            .dfa
                            .possible_future_group_ids(end_state)
                            .contains(step.terminal as usize)
                })
            {
                constraint.table.action(top_state, step.terminal)
            } else {
                None
            };
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), action_lookup_start) {
                profile.linear_fast_path_action_lookup_ns += start.elapsed().as_nanos() as u64;
            }
            if let Some(action) = carried_action {
                let apply_action_start = profile.as_ref().map(|_| std::time::Instant::now());
                if !template_advance_enabled()
                    && let Some(stack) = carried_stack.as_mut()
                {
                    match action {
                        Action::Shift(target, is_replace) => {
                            if *is_replace {
                                if stack.replace_top(*target) {
                                    shifted_carried_stack = true;
                                }
                            } else {
                                stack.push(*target);
                                shifted_carried_stack = true;
                            }
                        }
                        Action::StackShifts(shifts) => {
                            if let [shift] = shifts.as_slice()
                                && stack.pop(shift.pop as usize) == 0
                            {
                                for &target in &shift.pushes {
                                    stack.push(target);
                                }
                                shifted_carried_stack = true;
                            }
                        }
                        _ => {}
                    }
                }
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), apply_action_start) {
                    carried_apply_elapsed_ns = start.elapsed().as_nanos() as u64;
                    profile.linear_fast_path_apply_action_wall_ns += carried_apply_elapsed_ns;
                }
            }
            if shifted_carried_stack {
                if let Some(profile) = profile.as_deref_mut() {
                    let bookkeeping_start = std::time::Instant::now();
                    let advance_profile = fast_action_advance_profile(
                        &gss,
                        carried_action.unwrap(),
                        carried_apply_elapsed_ns,
                    );
                    profile.advance_core_ns += advance_profile.total_ns;
                    profile.advance_ns += carried_apply_elapsed_ns;
                    profile.linear_fast_path_advance_ns += carried_apply_elapsed_ns;
                    profile.n_advances += 1;
                    apply_advance_profile(profile, &advance_profile);
                    profile.linear_fast_path_profile_bookkeeping_ns +=
                        bookkeeping_start.elapsed().as_nanos() as u64;
                }

                offset += step.width;
                tokenizer_state = constraint.tokenizer.initial_state();
                continue;
            }

            if let Some(stack) = carried_stack.take() {
                let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
                gss = stack.into_gss();
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
                    profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
                }
            }
            let advance_start = profile.as_ref().map(|_| std::time::Instant::now());
            let advanced = if !template_advance_enabled()
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, step.terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(
                        &constraint.table,
                        &gss,
                        top_state,
                        step.terminal,
                        action,
                    )
            {
                advanced
            } else {
                if let Some(profile) = profile.as_deref_mut() {
                    let (advanced, advance_profile) =
                        advance_parser_stacks_profiled(constraint, &gss, step.terminal);
                    if advanced.is_empty() {
                        return None;
                    }
                    let bookkeeping_start = std::time::Instant::now();
                    profile.advance_core_ns += advance_profile.total_ns;
                    apply_advance_profile(profile, &advance_profile);
                    profile.linear_fast_path_profile_bookkeeping_ns +=
                        bookkeeping_start.elapsed().as_nanos() as u64;
                    advanced
                } else {
                    let advanced = advance_parser_stacks(constraint, &gss, step.terminal);
                    if advanced.is_empty() {
                        return None;
                    }
                    advanced
                }
            };
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), advance_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.linear_fast_path_apply_action_wall_ns += elapsed;
                profile.advance_ns += elapsed;
                profile.linear_fast_path_advance_ns += elapsed;
                profile.n_advances += 1;
            }
            if advanced.is_empty() {
                return Some(LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                )));
            }
            let exec_result = TokenizerExecResult {
                end_state: step.end_state,
                matches: Vec::new(),
            };
            let future_start = profile.as_ref().map(|_| std::time::Instant::now());
            gss = apply_future_terminal_disallow(
                constraint,
                &exec_result,
                step.terminal,
                advanced,
            );
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), future_start) {
                let elapsed = start.elapsed().as_nanos() as u64;
                profile.advance_future_disallow_ns += elapsed;
                profile.linear_fast_path_future_disallow_ns += elapsed;
                profile.linear_fast_path_advance_ns += elapsed;
            }
            if gss.is_empty() {
                return Some(LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                )));
            }
        }

        offset += step.width;
        tokenizer_state = constraint.tokenizer.initial_state();
    }

    if let Some(stack) = carried_stack.take() {
        let materialize_start = profile.as_ref().map(|_| std::time::Instant::now());
        gss = stack.into_gss();
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), materialize_start) {
            profile.linear_fast_path_materialize_ns += start.elapsed().as_nanos() as u64;
        }
    }
    let fuse_start = profile.as_ref().map(|_| std::time::Instant::now());
    let fused = gss.fuse(Some(1));
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), fuse_start) {
        let elapsed = start.elapsed().as_nanos() as u64;
        profile.linear_fast_path_fuse_ns += elapsed;
        profile.fuse_ns += elapsed;
    }
    if fused.is_empty() {
        return Some(LinearFastPathResult::Complete(Err(
            "commit rejected: no valid parser states remain".to_string(),
        )));
    }
    Some(LinearFastPathResult::Complete(Ok(fused)))
}

fn record_per_advance_entry(
    advances: &mut Vec<PerAdvanceEntry>,
    tokenizer_state: u32,
    terminal_id: u32,
    before_gss: &ParserGSS,
    after_gss: &ParserGSS,
    match_start: usize,
    match_end: usize,
    token_bound: usize,
    match_bytes: &[u8],
    profile: AdvanceProfile,
) -> u64 {
    use std::time::Instant;

    let summary_start = Instant::now();
    let gss_stacks_before = parser_stacks_only(before_gss);
    let gss_stacks_after = parser_stacks_only(after_gss);
    let gss_summary_before = before_gss.summary();
    let gss_summary_after = after_gss.summary();
    let match_bytes = match_bytes.to_vec();
    let summary_ns = summary_start.elapsed().as_nanos() as u64;
    advances.push(PerAdvanceEntry {
        terminal_id,
        tokenizer_state,
        gss_stacks_before,
        gss_stacks_after,
        gss_summary_before,
        gss_summary_after,
        match_start,
        match_end,
        token_bound,
        match_bytes,
        profile,
        summary_ns,
    });
    summary_ns
}

fn commit_bytes_fast_path_profiled(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
    tokenizer_state: u32,
    exec_result: &TokenizerExecResult,
    advances: Option<&mut Vec<PerAdvanceEntry>>,
    profile: &mut CommitProfile,
) -> Option<Result<(), String>> {
    use std::time::Instant;

    let total_start = Instant::now();
    let gss = state.values().next().unwrap();
    let ignore_terminal = constraint.ignore_terminal;

    let scan_start = Instant::now();
    let mut sole_terminal: Option<u32> = None;
    for matched in &exec_result.matches {
        if matched.width != bytes.len() {
            profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
            return None;
        }
        if is_ignored_terminal(ignore_terminal, matched.id) {
            profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
            return None;
        }
        if !stack_may_advance_on(&constraint.table, gss, matched.id) {
            continue;
        }
        if sole_terminal.is_some() {
            profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
            return None;
        }
        sole_terminal = Some(matched.id);
    }
    profile.fast_path_match_scan_ns = scan_start.elapsed().as_nanos() as u64;
    let Some(terminal) = sole_terminal else {
        profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
        return None;
    };

    let no_end_state = exec_result.end_state.is_none();
    let all_accs_empty = no_end_state
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());

    if all_accs_empty && !template_advance_enabled() {
        if let Some(top_state) = gss.single_exclusive_top_value() {
            if let Some(Action::Shift(target, is_replace)) = constraint.table.action(top_state, terminal) {
                let advance_start = Instant::now();
                let shifted = if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                };
                profile.fast_path_advance_ns = advance_start.elapsed().as_nanos() as u64;
                profile.advance_core_ns = profile.fast_path_advance_ns;
                profile.advance_ns = profile.fast_path_advance_ns;
                profile.n_advances = 1;

                if let Some(advances) = advances {
                    profile.adv_summary_ns += record_per_advance_entry(
                        advances,
                        tokenizer_state,
                        terminal,
                        gss,
                        &shifted,
                        0,
                        bytes.len(),
                        bytes.len(),
                        bytes,
                        AdvanceProfile {
                            pure_shift: true,
                            fast_path_ns: profile.fast_path_advance_ns,
                            stack_shift_apply_ns: profile.fast_path_advance_ns,
                            total_ns: profile.fast_path_advance_ns,
                            top_states: gss.peek_values().len() as u32,
                            gss_depth: gss.max_depth(),
                            vstack_len: gss.try_virtual_stack().map_or(0, |vstack| vstack.len() as u32),
                            ..AdvanceProfile::default()
                        },
                    );
                }

                let update_start = Instant::now();
                state.clear();
                state.insert(constraint.tokenizer.initial_state(), shifted);
                profile.fast_path_state_update_ns = update_start.elapsed().as_nanos() as u64;
                profile.fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                profile.total_ns = profile.fast_path_total_ns;
                profile.fast_path_tokenizer_exec_ns = profile.exec_ns;
                return Some(Ok(()));
            }
            if !template_advance_enabled()
                && let Some(Action::StackShifts(shifts)) = constraint.table.action(top_state, terminal)
            {
                let advance_start = Instant::now();
                let (shifted, advance_profile) =
                    advance_parser_stacks_profiled(constraint, gss, terminal);
                profile.fast_path_advance_ns = advance_start.elapsed().as_nanos() as u64;
                profile.advance_core_ns = profile.fast_path_advance_ns;
                profile.advance_ns = profile.fast_path_advance_ns;
                profile.n_advances = 1;
                apply_advance_profile(profile, &advance_profile);
                if let Some(advances) = advances {
                    profile.adv_summary_ns += record_per_advance_entry(
                        advances,
                        tokenizer_state,
                        terminal,
                        gss,
                        &shifted,
                        0,
                        bytes.len(),
                        bytes.len(),
                        bytes,
                        advance_profile,
                    );
                }

                let update_start = Instant::now();
                state.clear();
                state.insert(constraint.tokenizer.initial_state(), shifted);
                profile.fast_path_state_update_ns = update_start.elapsed().as_nanos() as u64;
                profile.fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                profile.total_ns = profile.fast_path_total_ns;
                profile.fast_path_tokenizer_exec_ns = profile.exec_ns;
                return Some(Ok(()));
            }
        }
    }

    let (_, gss_owned) = state.pop_first().unwrap();

    let prune_start = Instant::now();
    let pruned_gss = if all_accs_empty {
        gss_owned
    } else {
        let pruned = gss_owned.apply_and_prune_no_promote(|td: &TerminalsDisallowed| {
            if td.is_empty() {
                return Some(TerminalsDisallowed::new());
            }
            if let Some(disallowed) = td.get(&tokenizer_state) {
                if disallowed.contains(&terminal) {
                    return None;
                }
            }
            let mut remapped = BTreeMap::new();
            if let Some(end_state) = exec_result.end_state {
                if let Some(d) = td.get(&tokenizer_state) {
                    remapped
                        .entry(end_state)
                        .or_insert_with(BTreeSet::new)
                        .extend(d.iter().copied());
                }
            }
            Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
        });

        if pruned.is_empty() {
            return Some(Err("commit rejected: no valid parser states remain".to_string()));
        }
        pruned
    };
    profile.fast_path_prune_ns = prune_start.elapsed().as_nanos() as u64;
    profile.prune_ns = profile.fast_path_prune_ns;

    let end_state_check_start = Instant::now();
    let end_state_to_keep = exec_result
        .end_state
        .filter(|&end_state| end_state_may_advance(constraint, &pruned_gss, end_state));
    profile.fast_path_end_state_check_ns = end_state_check_start.elapsed().as_nanos() as u64;
    let end_state_gss = end_state_to_keep.map(|_| pruned_gss.clone());

    let advance_start = Instant::now();
    let advanced = advance_parser_stacks_owned(constraint, pruned_gss.clone(), terminal);
    profile.fast_path_advance_ns = advance_start.elapsed().as_nanos() as u64;
    profile.advance_core_ns = profile.fast_path_advance_ns;
    profile.n_advances = 1;

    if let Some(advances) = advances {
        let (after_for_entry, advance_profile) =
            advance_parser_stacks_profiled(constraint, &pruned_gss, terminal);
        profile.adv_summary_ns += record_per_advance_entry(
            advances,
            tokenizer_state,
            terminal,
            &pruned_gss,
            &after_for_entry,
            0,
            bytes.len(),
            bytes.len(),
            bytes,
            advance_profile.clone(),
        );
        apply_advance_profile(profile, &advance_profile);
    }

    let mut produced_state = false;
    if !advanced.is_empty() {
        let future_start = Instant::now();
        let advanced = apply_future_terminal_disallow(constraint, exec_result, terminal, advanced);
        profile.fast_path_future_disallow_ns = future_start.elapsed().as_nanos() as u64;
        profile.advance_future_disallow_ns = profile.fast_path_future_disallow_ns;
        profile.advance_ns = profile.fast_path_advance_ns + profile.fast_path_future_disallow_ns;

        if !advanced.is_empty() {
            let fuse_start = Instant::now();
            let fused = advanced.fuse(Some(1));
            profile.fast_path_fuse_ns = fuse_start.elapsed().as_nanos() as u64;
            profile.fuse_ns = profile.fast_path_fuse_ns;

            if !fused.is_empty() {
                let update_start = Instant::now();
                state.insert(constraint.tokenizer.initial_state(), fused);
                profile.fast_path_state_update_ns += update_start.elapsed().as_nanos() as u64;
                produced_state = true;
            }
        }
    } else {
        profile.advance_ns = profile.fast_path_advance_ns;
    }

    if let (Some(end_state), Some(end_gss)) = (end_state_to_keep, end_state_gss) {
        let fuse_start = Instant::now();
        let fused = end_gss.fuse(Some(1));
        let fuse_elapsed = fuse_start.elapsed().as_nanos() as u64;
        profile.fast_path_fuse_ns += fuse_elapsed;
        profile.fuse_ns += fuse_elapsed;
        if !fused.is_empty() {
            let update_start = Instant::now();
            state
                .entry(end_state)
                .and_modify(|existing| *existing = existing.merge(&fused))
                .or_insert(fused);
            profile.fast_path_state_update_ns += update_start.elapsed().as_nanos() as u64;
            produced_state = true;
        }
    }

    if !produced_state {
        return Some(Err("commit rejected: no valid parser states remain".to_string()));
    }
    profile.fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
    profile.total_ns = profile.fast_path_total_ns;
    profile.fast_path_tokenizer_exec_ns = profile.exec_ns;
    Some(Ok(()))
}

fn commit_bytes_impl_profiled(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    mut advances: Option<&mut Vec<PerAdvanceEntry>>,
    allow_fast_paths: bool,
) -> Result<CommitProfile, String> {
    use std::time::Instant;

    let total_start = Instant::now();
    let mut profile = CommitProfile {
        n_tokenizer_states: state.len() as u64,
        ..CommitProfile::default()
    };

    if bytes.is_empty() {
        profile.total_ns = total_start.elapsed().as_nanos() as u64;
        return Ok(profile);
    }

    let ignore_terminal = constraint.ignore_terminal;

    if allow_fast_paths && state.len() == 1 {
        let (&tokenizer_state, parser_gss) = state.iter().next().unwrap();
        if parser_gss.single_exclusive_top_value().is_some() {
            let direct_start = Instant::now();
            match commit_bytes_direct_linear_fast_path(
                constraint,
                parser_gss.clone(),
                bytes,
                tokenizer_state,
                Some(&mut profile),
            ) {
                Some(LinearFastPathResult::Complete(result)) => {
                    profile.linear_fast_path_total_ns = direct_start.elapsed().as_nanos() as u64;
                    let result = result.map(|final_gss| {
                        let update_start = Instant::now();
                        state.clear();
                        state.insert(constraint.tokenizer.initial_state(), final_gss);
                        profile.linear_fast_path_state_update_ns +=
                            update_start.elapsed().as_nanos() as u64;
                        profile.total_ns = total_start.elapsed().as_nanos() as u64;
                        profile
                    });
                    return result;
                }
                Some(LinearFastPathResult::Continue { .. }) => {
                    unreachable!("direct linear fast path never returns Continue")
                }
                Some(LinearFastPathResult::Restart) | None => {
                    profile.failed_fast_path_probe_ns += direct_start.elapsed().as_nanos() as u64;
                }
            }
        }

        let exec_start = Instant::now();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        let initial_exec_elapsed = exec_start.elapsed().as_nanos() as u64;
        profile.initial_exec_ns = initial_exec_elapsed;
        profile.exec_ns = initial_exec_elapsed;
        profile.fast_path_tokenizer_exec_ns = initial_exec_elapsed;

        if allow_fast_paths {
            if let Some(result) = commit_bytes_fast_path_profiled(
                constraint,
                state,
                bytes,
                tokenizer_state,
                &exec_result,
                advances.as_deref_mut(),
                &mut profile,
            ) {
                let result = result.map(|()| profile);
                return result;
            }

            let linear_eligibility_start = Instant::now();
            let linear_fast_path_eligible = !exec_result.end_state.is_some_and(|end_state| {
                    state
                        .values()
                        .next()
                        .is_some_and(|gss| end_state_may_advance(constraint, gss, end_state))
                });
            profile.linear_fast_path_eligibility_ns +=
                linear_eligibility_start.elapsed().as_nanos() as u64;
            if linear_fast_path_eligible {
                let linear_setup_start = Instant::now();
                let current_gss = state.values().next().unwrap();
                let start_gss = if current_gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
                    current_gss.clone()
                } else {
                    prune_single_initial_state_for_exec(
                        constraint,
                        current_gss.clone(),
                        tokenizer_state,
                        &exec_result,
                    )
                };
                if start_gss.is_empty() {
                    return Err("commit rejected: no valid parser states remain".to_string());
                }
                let mut linear_profile = profile.clone();
                let mut linear_advances = Vec::new();
                let linear_advances_sink = if advances.is_some() {
                    Some(&mut linear_advances)
                } else {
                    None
                };
                linear_profile.linear_fast_path_setup_ns +=
                    linear_setup_start.elapsed().as_nanos() as u64;
                match commit_bytes_linear_fast_path_profiled(
                    constraint,
                    start_gss,
                    bytes,
                    exec_result.clone(),
                    linear_advances_sink,
                    &mut linear_profile,
                ) {
                    LinearFastPathResult::Complete(result) => {
                        let result = result.map(|final_gss| {
                            if let Some(advances) = advances.as_deref_mut() {
                                advances.extend(linear_advances);
                            }
                            let update_start = Instant::now();
                            state.clear();
                            state.insert(constraint.tokenizer.initial_state(), final_gss);
                            linear_profile.linear_fast_path_state_update_ns +=
                                update_start.elapsed().as_nanos() as u64;
                            linear_profile.total_ns = total_start.elapsed().as_nanos() as u64;
                            linear_profile
                        });
                        return result;
                    }
                    LinearFastPathResult::Continue { gss, offset } => {
                        profile = linear_profile;
                        if let Some(advances) = advances.as_deref_mut() {
                            advances.extend(linear_advances);
                        }
                        let update_start = Instant::now();
                        state.clear();
                        state.insert(constraint.tokenizer.initial_state(), gss);
                        profile.linear_fast_path_state_update_ns +=
                            update_start.elapsed().as_nanos() as u64;

                        let queue_start = Instant::now();
                        let needed_queue_len = bytes.len() + 1;
                        let mut pending_state = ParserStatesByTokenizer::default();
                        let mut processing_queue: Vec<ParserStatesByTokenizer> =
                            (0..needed_queue_len).map(|_| ParserStatesByTokenizer::default()).collect();
                        processing_queue[offset] = std::mem::take(state).into_iter().collect();

                        let mut queue_offset = offset;
                        while queue_offset < needed_queue_len {
                            if processing_queue[queue_offset].is_empty() {
                                queue_offset += 1;
                                continue;
                            }

                            let states_to_process = std::mem::take(&mut processing_queue[queue_offset]);
                            for (tokenizer_state, gss_at_offset) in states_to_process {
                                profile.n_queue_entries += 1;

                                let actionable_start = Instant::now();
                                let actionable_terminals =
                                    ActionableTerminals::from_gss(constraint, &gss_at_offset);
                                profile.actionable_ns += actionable_start.elapsed().as_nanos() as u64;

                                let exec_start = Instant::now();
                                let exec_result = execute_tokenizer_from_state_small(
                                    constraint,
                                    &bytes[queue_offset..],
                                    tokenizer_state,
                                );
                                let queue_exec_elapsed = exec_start.elapsed().as_nanos() as u64;
                                profile.queue_exec_ns += queue_exec_elapsed;
                                profile.exec_ns += queue_exec_elapsed;

                                let match_start = Instant::now();
                                let normalized_matches = collect_unique_actionable_matches(
                                    constraint,
                                    actionable_terminals.as_ref(),
                                    ignore_terminal,
                                    &exec_result.matches,
                                    None,
                                );
                                profile.queue_match_ns += match_start.elapsed().as_nanos() as u64;

                                for matched in normalized_matches {
                                    let new_offset = queue_offset + matched.width;

                                    if matched.ignored {
                                        let enqueue_start = Instant::now();
                                        queue_parser_state(
                                            &mut processing_queue,
                                            &mut pending_state,
                                            new_offset,
                                            bytes.len(),
                                            constraint.tokenizer.initial_state(),
                                            gss_at_offset.clone(),
                                        );
                                        profile.queue_enqueue_ns +=
                                            enqueue_start.elapsed().as_nanos() as u64;
                                        continue;
                                    }

                                    let may_start = Instant::now();
                                    let may_advance =
                                        stack_may_advance_on(&constraint.table, &gss_at_offset, matched.terminal_id);
                                    let may_elapsed = may_start.elapsed().as_nanos() as u64;
                                    profile.advance_may_check_ns += may_elapsed;
                                    if !may_advance {
                                        continue;
                                    }

                                    let advance_core_start = Instant::now();
                                    let (advanced_before_disallow, advance_profile) =
                                        advance_parser_stacks_profiled(
                                            constraint,
                                            &gss_at_offset,
                                            matched.terminal_id,
                                        );
                                    let advance_core_elapsed =
                                        advance_core_start.elapsed().as_nanos() as u64;
                                    profile.advance_core_ns += advance_core_elapsed;
                                    apply_advance_profile(&mut profile, &advance_profile);

                                    if let Some(advances) = advances.as_deref_mut() {
                                        profile.adv_summary_ns += record_per_advance_entry(
                                            advances,
                                            tokenizer_state,
                                            matched.terminal_id,
                                            &gss_at_offset,
                                            &advanced_before_disallow,
                                            queue_offset,
                                            new_offset,
                                            bytes.len(),
                                            &bytes[queue_offset..new_offset],
                                            advance_profile.clone(),
                                        );
                                    }

                                    let future_start = Instant::now();
                                    let advanced = apply_future_terminal_disallow(
                                        constraint,
                                        &exec_result,
                                        matched.terminal_id,
                                        advanced_before_disallow,
                                    );
                                    let future_elapsed = future_start.elapsed().as_nanos() as u64;
                                    profile.advance_future_disallow_ns += future_elapsed;
                                    profile.advance_ns +=
                                        may_elapsed + advance_core_elapsed + future_elapsed;
                                    profile.n_advances += 1;

                                    if advanced.is_empty() {
                                        continue;
                                    }

                                    let enqueue_start = Instant::now();
                                    queue_parser_state(
                                        &mut processing_queue,
                                        &mut pending_state,
                                        new_offset,
                                        bytes.len(),
                                        constraint.tokenizer.initial_state(),
                                        advanced,
                                    );
                                    profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
                                }

                                if let Some(end_state) = exec_result.end_state {
                                    let may_start = Instant::now();
                                    let may_advance =
                                        end_state_may_advance(constraint, &gss_at_offset, end_state);
                                    profile.may_advance_ns += may_start.elapsed().as_nanos() as u64;
                                    if !may_advance {
                                        continue;
                                    }

                                    let enqueue_start = Instant::now();
                                    queue_parser_state(
                                        &mut processing_queue,
                                        &mut pending_state,
                                        bytes.len(),
                                        bytes.len(),
                                        end_state,
                                        gss_at_offset,
                                    );
                                    profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
                                }
                            }
                            queue_offset += 1;

                        }

                        profile.queue_ns = queue_start.elapsed().as_nanos() as u64;
                        let queue_accounted_ns = profile
                            .actionable_ns
                            .saturating_add(profile.queue_exec_ns)
                            .saturating_add(profile.queue_match_ns)
                            .saturating_add(profile.advance_ns)
                            .saturating_add(profile.may_advance_ns)
                            .saturating_add(profile.queue_enqueue_ns);
                        profile.queue_bookkeeping_ns =
                            profile.queue_ns.saturating_sub(queue_accounted_ns);

                        let fuse_start = Instant::now();
                        let new_state = finalize_pending_state(std::mem::take(&mut pending_state));
                        profile.fuse_ns += fuse_start.elapsed().as_nanos() as u64;

                        *state = new_state;
                        if state.is_empty() {
                            return Err("commit rejected: no valid parser states remain".to_string());
                        }

                        profile.total_ns = total_start.elapsed().as_nanos() as u64;
                        return Ok(profile);
                    }
                    LinearFastPathResult::Restart => {
                        profile = linear_profile;
                    }
                }
            }
        }
    }

    let scan_start = Instant::now();
    let mut initial_scan = InitialCommitScan::collect(constraint, state, bytes);
    profile.scan_ns = scan_start.elapsed().as_nanos() as u64;

    let prune_start = Instant::now();
    prune_initial_states(
        state,
        &initial_scan.accepted_terminals,
        &initial_scan.remapped_tokenizer_states,
    );
    state.retain(|_, parser_state| !parser_state.is_empty());
    profile.prune_ns = prune_start.elapsed().as_nanos() as u64;

    let queue_start = Instant::now();
    let mut pending_state = ParserStatesByTokenizer::default();
    let mut processing_queue: Vec<ParserStatesByTokenizer> =
        (0..=bytes.len()).map(|_| ParserStatesByTokenizer::default()).collect();
    processing_queue[0] = std::mem::take(state).into_iter().collect();

    let mut offset = 0usize;
    while offset < processing_queue.len() {
        if processing_queue[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut processing_queue[offset]);
        for (tokenizer_state, gss_at_offset) in states_to_process {
            profile.n_queue_entries += 1;

            let actionable_start = Instant::now();
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            profile.actionable_ns += actionable_start.elapsed().as_nanos() as u64;

            let exec_start = Instant::now();
            let exec_result = if offset == 0 {
                initial_scan.take_exec_result(tokenizer_state).unwrap_or_else(|| {
                    execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
                })
            } else {
                execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
            };
            let queue_exec_elapsed = exec_start.elapsed().as_nanos() as u64;
            profile.queue_exec_ns += queue_exec_elapsed;
            profile.exec_ns += queue_exec_elapsed;

            let match_start = Instant::now();
            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                actionable_terminals.as_ref(),
                ignore_terminal,
                &exec_result.matches,
                None,
            );
            profile.queue_match_ns += match_start.elapsed().as_nanos() as u64;

            for matched in normalized_matches {
                let new_offset = offset + matched.width;

                if matched.ignored {
                    let enqueue_start = Instant::now();
                    queue_parser_state(
                        &mut processing_queue,
                        &mut pending_state,
                        new_offset,
                        bytes.len(),
                        constraint.tokenizer.initial_state(),
                        gss_at_offset.clone(),
                    );
                    profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
                    continue;
                }

                let may_start = Instant::now();
                let may_advance =
                    stack_may_advance_on(&constraint.table, &gss_at_offset, matched.terminal_id);
                let may_elapsed = may_start.elapsed().as_nanos() as u64;
                profile.advance_may_check_ns += may_elapsed;
                if !may_advance {
                    continue;
                }

                let advance_core_start = Instant::now();
                let (advanced_before_disallow, advance_profile) =
                    advance_parser_stacks_profiled(
                        constraint,
                        &gss_at_offset,
                        matched.terminal_id,
                    );
                let advance_core_elapsed = advance_core_start.elapsed().as_nanos() as u64;
                profile.advance_core_ns += advance_core_elapsed;
                apply_advance_profile(&mut profile, &advance_profile);

                if let Some(advances) = advances.as_deref_mut() {
                    profile.adv_summary_ns += record_per_advance_entry(
                        advances,
                        tokenizer_state,
                        matched.terminal_id,
                        &gss_at_offset,
                        &advanced_before_disallow,
                        offset,
                        new_offset,
                        bytes.len(),
                        &bytes[offset..new_offset],
                        advance_profile.clone(),
                    );
                }

                let future_start = Instant::now();
                let advanced = apply_future_terminal_disallow(
                    constraint,
                    &exec_result,
                    matched.terminal_id,
                    advanced_before_disallow,
                );
                let future_elapsed = future_start.elapsed().as_nanos() as u64;
                profile.advance_future_disallow_ns += future_elapsed;
                profile.advance_ns += may_elapsed + advance_core_elapsed + future_elapsed;
                profile.n_advances += 1;

                if advanced.is_empty() {
                    continue;
                }

                let enqueue_start = Instant::now();
                queue_parser_state(
                    &mut processing_queue,
                    &mut pending_state,
                    new_offset,
                    bytes.len(),
                    constraint.tokenizer.initial_state(),
                    advanced,
                );
                profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
            }

            if let Some(end_state) = exec_result.end_state {
                let may_start = Instant::now();
                let may_advance = end_state_may_advance(constraint, &gss_at_offset, end_state);
                profile.may_advance_ns += may_start.elapsed().as_nanos() as u64;
                if !may_advance {
                    continue;
                }

                let enqueue_start = Instant::now();
                queue_parser_state(
                    &mut processing_queue,
                    &mut pending_state,
                    bytes.len(),
                    bytes.len(),
                    end_state,
                    gss_at_offset,
                );
                profile.queue_enqueue_ns += enqueue_start.elapsed().as_nanos() as u64;
            }
        }
        offset += 1;
    }
    profile.queue_ns = queue_start.elapsed().as_nanos() as u64;
    let queue_accounted_ns = profile
        .actionable_ns
        .saturating_add(profile.queue_exec_ns)
        .saturating_add(profile.queue_match_ns)
        .saturating_add(profile.advance_ns)
        .saturating_add(profile.may_advance_ns)
        .saturating_add(profile.queue_enqueue_ns);
    profile.queue_bookkeeping_ns = profile.queue_ns.saturating_sub(queue_accounted_ns);

    let fuse_start = Instant::now();

    let new_state = finalize_pending_state(std::mem::take(&mut pending_state));
    profile.fuse_ns = fuse_start.elapsed().as_nanos() as u64;

    *state = new_state;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    profile.total_ns = total_start.elapsed().as_nanos() as u64;
    Ok(profile)
}

fn final_stacks(state: &BTreeMap<u32, ParserGSS>) -> Vec<(u32, Vec<Vec<u32>>)> {
    state.iter().map(|(&tokenizer_state, gss)| {
        (tokenizer_state, parser_stacks_only(gss))
    }).collect()
}

fn commit_bytes_linear_fast_path(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    first_exec_result: TokenizerExecResult,
) -> LinearFastPathResult {
    let ignore_terminal = constraint.ignore_terminal;
    let mut gss = start_gss;
    let mut carried_stack = gss.try_virtual_stack();
    let mut offset = 0usize;
    let mut exec_result = first_exec_result;

    loop {
        let actionable_terminals = if let Some(stack) = carried_stack.as_ref() {
            stack.top().copied().map(ActionableTerminals::SingleState)
        } else {
            ActionableTerminals::from_gss(constraint, &gss)
        };
        let mut chosen: Option<(usize, u32, bool)> = None;

        for matched in &exec_result.matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored
                && !is_actionable_terminal(
                    actionable_terminals.as_ref(),
                    constraint,
                    matched.id,
                )
            {
                continue;
            }

            let candidate = (matched.width, matched.id, ignored);
            if let Some(existing) = chosen {
                if existing != candidate {
                    return if offset > 0 {
                        LinearFastPathResult::Continue { gss, offset }
                    } else {
                        LinearFastPathResult::Restart
                    };
                }
            } else {
                chosen = Some(candidate);
            }
        }

        let Some((width, terminal, ignored)) = chosen else {
            return if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
        };

        if let Some(end_state) = exec_result.end_state
            && let Some(stack) = carried_stack.as_ref()
        {
            let keep_carried = stack.top().copied().is_some_and(|top_state| {
                end_state != constraint.tokenizer.initial_state()
                    && !constraint.table.advance_row_intersects(
                        top_state,
                        constraint.tokenizer.possible_future_terminals(end_state),
                    )
                    && !constraint
                        .tokenizer
                        .dfa
                        .possible_future_group_ids(end_state)
                        .contains(terminal as usize)
            });
            if !keep_carried {
                gss = carried_stack.take().unwrap().into_gss();
            }
        }

        if let Some(end_state) = exec_result.end_state {
            if end_state_may_advance(constraint, &gss, end_state) {
                return if offset > 0 {
                    LinearFastPathResult::Continue { gss, offset }
                } else {
                    LinearFastPathResult::Restart
                };
            }
        }

        if !ignored {
            let mut shifted_carried_stack = false;
            if !template_advance_enabled()
                && let Some(stack) = carried_stack.as_mut()
                && let Some(top_state) = stack.top().copied()
                && let Some(Action::Shift(target, is_replace)) = constraint.table.action(top_state, terminal)
                && exec_result.end_state.is_none_or(|end_state| {
                    end_state != constraint.tokenizer.initial_state()
                        && !constraint.table.advance_row_intersects(
                            top_state,
                            constraint.tokenizer.possible_future_terminals(end_state),
                        )
                        && !constraint
                            .tokenizer
                            .dfa
                            .possible_future_group_ids(end_state)
                            .contains(terminal as usize)
                })
            {
                if *is_replace {
                    if stack.replace_top(*target) {
                        shifted_carried_stack = true;
                    }
                } else {
                    stack.push(*target);
                    shifted_carried_stack = true;
                }
            }

            if shifted_carried_stack {
                offset += width;
                if offset == bytes.len() {
                    gss = carried_stack.take().unwrap().into_gss();
                    let fused = gss.fuse(Some(1));
                    if fused.is_empty() {
                        return LinearFastPathResult::Complete(Err(
                            "commit rejected: no valid parser states remain".to_string(),
                        ));
                    }
                    return LinearFastPathResult::Complete(Ok(fused));
                }

                exec_result = execute_tokenizer_from_state_small(
                    constraint,
                    &bytes[offset..],
                    constraint.tokenizer.initial_state(),
                );
                continue;
            }

            if let Some(stack) = carried_stack.take() {
                gss = stack.into_gss();
            }

            let fast_advanced = if !template_advance_enabled()
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
            {
                apply_single_top_action_fast(&constraint.table, &gss, top_state, terminal, action)
            } else {
                None
            };

            let advanced = if let Some(advanced) = fast_advanced {
                advanced
            } else {
                let advanced = advance_parser_stacks(constraint, &gss, terminal);
                if advanced.is_empty() {
                    return if offset > 0 {
                        LinearFastPathResult::Continue { gss, offset }
                    } else {
                        LinearFastPathResult::Restart
                    };
                }
                advanced
            };
            if advanced.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            gss = apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
            if gss.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
        }

        offset += width;
        if offset == bytes.len() {
            if let Some(stack) = carried_stack.take() {
                gss = stack.into_gss();
            }
            let fused = gss.fuse(Some(1));
            if fused.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            return LinearFastPathResult::Complete(Ok(fused));
        }

        exec_result = execute_tokenizer_from_state_small(
            constraint,
            &bytes[offset..],
            constraint.tokenizer.initial_state(),
        );
    }
}

fn commit_bytes_linear_fast_path_profiled(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    first_exec_result: TokenizerExecResult,
    mut advances: Option<&mut Vec<PerAdvanceEntry>>,
    profile: &mut CommitProfile,
) -> LinearFastPathResult {
    use std::time::Instant;

    let total_start = Instant::now();
    let ignore_terminal = constraint.ignore_terminal;
    let mut gss = start_gss;
    let mut offset = 0usize;
    let mut exec_result = first_exec_result;
    profile.linear_fast_path_exec_ns = profile.initial_exec_ns;

    loop {
        profile.linear_fast_path_steps += 1;

        let scan_start = Instant::now();
        let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss);
        let mut chosen: Option<(usize, u32, bool)> = None;
        for matched in &exec_result.matches {
            let ignored = is_ignored_terminal(ignore_terminal, matched.id);
            if !ignored
                && !is_actionable_terminal(
                    actionable_terminals.as_ref(),
                    constraint,
                    matched.id,
                )
            {
                continue;
            }

            let candidate = (matched.width, matched.id, ignored);
            if let Some(existing) = chosen {
                if existing != candidate {
                    let result = if offset > 0 {
                        LinearFastPathResult::Continue { gss, offset }
                    } else {
                        LinearFastPathResult::Restart
                    };
                    profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                    return result;
                }
            } else {
                chosen = Some(candidate);
            }
        }
        profile.linear_fast_path_match_scan_ns += scan_start.elapsed().as_nanos() as u64;

        let Some((width, terminal, ignored)) = chosen else {
            let result = if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
            profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
            return result;
        };

        let end_state_start = Instant::now();
        if let Some(end_state) = exec_result.end_state {
            if end_state_may_advance(constraint, &gss, end_state) {
                profile.linear_fast_path_end_state_check_ns +=
                    end_state_start.elapsed().as_nanos() as u64;
                let result = if offset > 0 {
                    LinearFastPathResult::Continue { gss, offset }
                } else {
                    LinearFastPathResult::Restart
                };
                profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                return result;
            }
        }
        profile.linear_fast_path_end_state_check_ns +=
            end_state_start.elapsed().as_nanos() as u64;

        if !ignored {
            let fast_start = Instant::now();
            let fast_advanced = if !template_advance_enabled()
                && let Some(top_state) = gss.single_exclusive_top_value()
                && let Some(action) = constraint.table.action(top_state, terminal)
                && let Some(advanced) =
                    apply_single_top_action_fast(
                        &constraint.table,
                        &gss,
                        top_state,
                        terminal,
                        action,
                    )
            {
                let elapsed = fast_start.elapsed().as_nanos() as u64;
                Some((advanced, fast_action_advance_profile(&gss, action, elapsed)))
            } else {
                None
            };

            let (advanced, advance_profile, advance_elapsed) =
                if let Some((advanced, advance_profile)) = fast_advanced {
                    let advance_elapsed = advance_profile.total_ns;
                    (advanced, advance_profile, advance_elapsed)
                } else {
                    let advance_start = Instant::now();
                    let (advanced, advance_profile) =
                        advance_parser_stacks_profiled(constraint, &gss, terminal);
                    let advance_elapsed = advance_start.elapsed().as_nanos() as u64;
                    if advanced.is_empty() {
                        profile.advance_core_ns += advance_elapsed;
                        let result = if offset > 0 {
                            LinearFastPathResult::Continue { gss, offset }
                        } else {
                            LinearFastPathResult::Restart
                        };
                        profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                        return result;
                    }
                    (advanced, advance_profile, advance_elapsed)
                };
            profile.advance_core_ns += advance_profile.total_ns;
            profile.linear_fast_path_advance_ns += advance_profile.total_ns;
            apply_advance_profile(profile, &advance_profile);

            if advanced.is_empty() {
                profile.advance_ns += advance_elapsed;
                profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }

            if let Some(advances) = advances.as_deref_mut() {
                let summary_ns = record_per_advance_entry(
                    advances,
                    constraint.tokenizer.initial_state(),
                    terminal,
                    &gss,
                    &advanced,
                    offset,
                    offset + width,
                    bytes.len(),
                    &bytes[offset..offset + width],
                    advance_profile.clone(),
                );
                profile.adv_summary_ns += summary_ns;
            }

            let future_start = Instant::now();
            gss = apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
            let future_elapsed = future_start.elapsed().as_nanos() as u64;
            profile.advance_future_disallow_ns += future_elapsed;
            profile.linear_fast_path_future_disallow_ns += future_elapsed;
            profile.linear_fast_path_advance_ns += future_elapsed;
            profile.advance_ns += advance_elapsed + future_elapsed;
            profile.n_advances += 1;
            if gss.is_empty() {
                profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
        }

        offset += width;
        if offset == bytes.len() {
            let fuse_start = Instant::now();
            let fused = gss.fuse(Some(1));
            profile.linear_fast_path_fuse_ns = fuse_start.elapsed().as_nanos() as u64;
            profile.fuse_ns = profile.linear_fast_path_fuse_ns;
            profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
            if fused.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            return LinearFastPathResult::Complete(Ok(fused));
        }

        let exec_start = Instant::now();
        exec_result = execute_tokenizer_from_state_small(
            constraint,
            &bytes[offset..],
            constraint.tokenizer.initial_state(),
        );
        let exec_elapsed = exec_start.elapsed().as_nanos() as u64;
        profile.linear_fast_path_exec_ns += exec_elapsed;
        profile.exec_ns += exec_elapsed;
    }
}

fn commit_bytes_impl(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }

    let ignore_terminal = constraint.ignore_terminal;

    if state.len() == 1 {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        if let Some(result) = commit_bytes_fast_path(
            constraint,
            state,
            bytes,
            tokenizer_state,
            &exec_result,
        ) {
            return result;
        }
    }

    // Single tokenizer state: execute tokenizer ONCE, try fast path, reuse result
    if state.len() == 1 {
        let (&tokenizer_state, parser_gss) = state.iter().next().unwrap();
        if parser_gss.single_exclusive_top_value().is_some() {
            if let Some(result) = commit_bytes_direct_linear_fast_path(
                constraint,
                parser_gss.clone(),
                bytes,
                tokenizer_state,
                None,
            ) {
                match result {
                    LinearFastPathResult::Complete(result) => match result {
                        Ok(final_gss) => {
                            state.clear();
                            state.insert(constraint.tokenizer.initial_state(), final_gss);
                            return Ok(());
                        }
                        Err(err) => return Err(err),
                    },
                    LinearFastPathResult::Continue { gss, offset } => {
                        state.clear();
                        state.insert(constraint.tokenizer.initial_state(), gss);
                        return commit_bytes_impl(constraint, state, &bytes[offset..], bufs);
                    }
                    LinearFastPathResult::Restart => {}
                }
            }
        }
    }

    if let Some(result) = commit_bytes_small_queue_fast_path(constraint, state, bytes) {
        return result;
    }

    if let Some(result) = commit_bytes_full_width_fast_path(constraint, state, bytes) {
        return result;
    }

    if state.len() == 1 {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);

        // Try fast path with pre-computed exec_result
        if let Some(result) = commit_bytes_fast_path(
            constraint, state, bytes, tokenizer_state, &exec_result,
        ) {
            return result;
        }

        if !exec_result
            .end_state
            .is_some_and(|end_state| {
                state
                    .values()
                    .next()
                    .is_some_and(|gss| end_state_may_advance(constraint, gss, end_state))
            })
        {
            let current_gss = state.values().next().unwrap();
            let start_gss = if current_gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()) {
                current_gss.clone()
            } else {
                prune_single_initial_state_for_exec(
                    constraint,
                    current_gss.clone(),
                    tokenizer_state,
                    &exec_result,
                )
            };
            if start_gss.is_empty() {
                return Err("commit rejected: no valid parser states remain".to_string());
            }
            match commit_bytes_linear_fast_path(
                constraint,
                start_gss,
                bytes,
                exec_result.clone(),
            ) {
                LinearFastPathResult::Complete(result) => {
                    match result {
                        Ok(final_gss) => {
                            state.clear();
                            state.insert(constraint.tokenizer.initial_state(), final_gss);
                            return Ok(());
                        }
                        Err(err) => return Err(err),
                    }
                }
                LinearFastPathResult::Continue { gss, offset } => {
                    bufs.clear_all();
                    state.clear();
                    state.insert(constraint.tokenizer.initial_state(), gss);

                    if bytes.len() - offset == 1 {
                        return commit_bytes_impl(constraint, state, &bytes[offset..], bufs);
                    }

                    let needed_queue_len = bytes.len() + 1;
                    let mut processing_queue = std::mem::take(&mut bufs.processing_queue);
                    if processing_queue.len() < needed_queue_len {
                        processing_queue.resize_with(needed_queue_len, ParserStatesByTokenizer::default);
                    }
                    for bucket in processing_queue.iter_mut().take(needed_queue_len) {
                        bucket.clear();
                    }
                    processing_queue[offset] = std::mem::take(state).into_iter().collect();

                    let mut offset = offset;
                    while offset < needed_queue_len {
                        if processing_queue[offset].is_empty() {
                            offset += 1;
                            continue;
                        }

                        let states_to_process = std::mem::take(&mut processing_queue[offset]);
                        for (tokenizer_state, gss_at_offset) in states_to_process {
                            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
                            let exec_result = execute_tokenizer_from_state_small(
                                constraint,
                                &bytes[offset..],
                                tokenizer_state,
                            );

                            bufs.terminal_result_cache.clear();

                            let normalized_matches = collect_unique_actionable_matches(
                                constraint,
                                actionable_terminals.as_ref(),
                                ignore_terminal,
                                &exec_result.matches,
                                Some(&mut bufs.seen_matches),
                            );

                            for matched in normalized_matches {
                                let new_offset = offset + matched.width;

                                if matched.ignored {
                                    queue_parser_state(
                                        &mut processing_queue,
                                        &mut bufs.pending_state,
                                        new_offset,
                                        bytes.len(),
                                        constraint.tokenizer.initial_state(),
                                        gss_at_offset.clone(),
                                    );
                                    continue;
                                }

                                let Some(gss) = advance_terminal_match(
                                    constraint,
                                    &gss_at_offset,
                                    matched.terminal_id,
                                    &exec_result,
                                    &mut bufs.advance_result_cache,
                                    &mut bufs.terminal_result_cache,
                                ) else {
                                    continue;
                                };

                                queue_parser_state(
                                    &mut processing_queue,
                                    &mut bufs.pending_state,
                                    new_offset,
                                    bytes.len(),
                                    constraint.tokenizer.initial_state(),
                                    gss,
                                );
                            }

                            if let Some(end_state) = exec_result.end_state {
                                if !end_state_may_advance(constraint, &gss_at_offset, end_state) {
                                    continue;
                                }

                                queue_parser_state(
                                    &mut processing_queue,
                                    &mut bufs.pending_state,
                                    bytes.len(),
                                    bytes.len(),
                                    end_state,
                                    gss_at_offset,
                                );
                            }
                        }
                    }

                    let new_state = finalize_pending_state(std::mem::take(&mut bufs.pending_state));

                    *state = new_state;
                    bufs.processing_queue = processing_queue;
                    if state.is_empty() {
                        return Err("commit rejected: no valid parser states remain".to_string());
                    }
                    return Ok(());
                }
                LinearFastPathResult::Restart => {}
            }
        }

        // Fast path failed — build scan data from already-computed exec_result
        bufs.clear_all();
        let parser_gss = state.values().next().unwrap();
        let actionable_terminals = ActionableTerminals::from_gss(constraint, parser_gss);

        if let Some(end_state) = exec_result.end_state {
            bufs.remapped_tokenizer_states.insert(tokenizer_state, end_state);
        }

        for matched in &exec_result.matches {
            if is_ignored_terminal(ignore_terminal, matched.id)
                || !is_actionable_terminal(
                    actionable_terminals.as_ref(),
                    constraint,
                    matched.id,
                )
            {
                continue;
            }

            bufs.accepted_terminals
                .entry(tokenizer_state)
                .or_default()
                .insert(matched.id);
        }

        bufs.exec_results.insert(tokenizer_state, exec_result);
    } else {
        bufs.clear_all();

        for (&tokenizer_state, parser_gss) in state.iter() {
            let actionable_terminals = ActionableTerminals::from_gss(constraint, parser_gss);
            let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);

            if let Some(end_state) = exec_result.end_state {
                bufs.remapped_tokenizer_states.insert(tokenizer_state, end_state);
            }

            for matched in &exec_result.matches {
                if is_ignored_terminal(ignore_terminal, matched.id)
                    || !is_actionable_terminal(
                        actionable_terminals.as_ref(),
                        constraint,
                        matched.id,
                    )
                {
                    continue;
                }

                bufs.accepted_terminals
                    .entry(tokenizer_state)
                    .or_default()
                    .insert(matched.id);
            }

            bufs.exec_results.insert(tokenizer_state, exec_result);
        }
    }

    prune_initial_states(
        state,
        &bufs.accepted_terminals,
        &bufs.remapped_tokenizer_states,
    );

    state.retain(|_, parser_state| !parser_state.is_empty());

    let needed_queue_len = bytes.len() + 1;
    let mut processing_queue = std::mem::take(&mut bufs.processing_queue);
    if processing_queue.len() < needed_queue_len {
        processing_queue.resize_with(needed_queue_len, ParserStatesByTokenizer::default);
    }
    for bucket in processing_queue.iter_mut().take(needed_queue_len) {
        bucket.clear();
    }
    processing_queue[0] = std::mem::take(state).into_iter().collect();

    let mut offset = 0usize;
    while offset < needed_queue_len {
        if processing_queue[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut processing_queue[offset]);
        for (tokenizer_state, gss_at_offset) in states_to_process {
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            let exec_result = if offset == 0 {
                bufs.exec_results.remove(&tokenizer_state).unwrap_or_else(|| {
                    execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
                })
            } else {
                execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
            };

            bufs.terminal_result_cache.clear();

            let normalized_matches = collect_unique_actionable_matches(
                constraint,
                actionable_terminals.as_ref(),
                ignore_terminal,
                &exec_result.matches,
                Some(&mut bufs.seen_matches),
            );

            for matched in normalized_matches {
                let new_offset = offset + matched.width;

                if matched.ignored {
                    queue_parser_state(
                        &mut processing_queue,
                        &mut bufs.pending_state,
                        new_offset,
                        bytes.len(),
                        constraint.tokenizer.initial_state(),
                        gss_at_offset.clone(),
                    );
                    continue;
                }

                let Some(gss) = advance_terminal_match(
                    constraint,
                    &gss_at_offset,
                    matched.terminal_id,
                    &exec_result,
                    &mut bufs.advance_result_cache,
                    &mut bufs.terminal_result_cache,
                ) else {
                    continue;
                };

                queue_parser_state(
                    &mut processing_queue,
                    &mut bufs.pending_state,
                    new_offset,
                    bytes.len(),
                    constraint.tokenizer.initial_state(),
                    gss,
                );
            }

            if let Some(end_state) = exec_result.end_state {
                if !end_state_may_advance(constraint, &gss_at_offset, end_state) {
                    continue;
                }

                queue_parser_state(
                    &mut processing_queue,
                    &mut bufs.pending_state,
                    bytes.len(),
                    bytes.len(),
                    end_state,
                    gss_at_offset,
                );
            }
        }

        offset += 1;
    }

    let new_state = finalize_pending_state(std::mem::take(&mut bufs.pending_state));

    *state = new_state;
    bufs.processing_queue = processing_queue;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    Ok(())
}

impl<'a> ConstraintState<'a> {
    /// Commit a sampled token, advancing the constraint state.
    ///
    /// `token_id` must be a token that exists in the vocabulary the constraint
    /// was built with.  Committing a token that is grammatically invalid (not
    /// in the current mask) drives the constraint into a fail state — this is
    /// normal and observable via an all-zero mask.
    ///
    /// # Errors
    ///
    /// Returns an error if `token_id` is not present in the vocabulary at all.
    pub fn commit_token(
        &mut self,
        token_id: u32,
    ) -> Result<(), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let result = commit_bytes_impl(constraint, &mut self.state, bytes, &mut self.buffers);
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    pub fn commit_token_timed_ns(&mut self, token_id: u32) -> Result<u64, String> {
        use std::time::Instant;

        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| format!("commit_token: token_id {token_id} not in vocabulary"))?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let start = Instant::now();
        let result = commit_bytes_impl(constraint, &mut self.state, bytes, &mut self.buffers);
        let total_ns = start.elapsed().as_nanos() as u64;
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result.map(|()| total_ns)
    }

    pub fn commit_token_profiled(&mut self, token_id: u32) -> Result<CommitProfile, String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| format!("commit_token: token_id {token_id} not in vocabulary"))?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let result = commit_bytes_impl_profiled(
            constraint,
            &mut self.state,
            bytes,
            &mut self.buffers,
            None,
            true,
        );
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    pub fn commit_token_per_advance(
        &mut self,
        token_id: u32,
    ) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| format!("commit_token: token_id {token_id} not in vocabulary"))?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let mut advances = Vec::new();
        let result = commit_bytes_impl_profiled(
            constraint,
            &mut self.state,
            bytes,
            &mut self.buffers,
            Some(&mut advances),
            profile_allow_fast_paths(),
        )
        .map(|profile| (advances, final_stacks(&self.state), profile));
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let result = commit_bytes_impl(self.constraint, &mut self.state, bytes, &mut self.buffers);
        self.generation += 1;
        result
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token(token)?;
        }
        Ok(())
    }
}
