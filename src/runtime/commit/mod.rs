use crate::automata::lexer::Lexer;
pub(crate) mod profile;
mod template_advance;
pub(crate) mod tokenizer_scan;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::automata::lexer::tokenizer::{TokenizerExecResult, TokenizerMatch, TokenizerStateSet};
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
use crate::ds::bitset::BitSet;
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

fn format_optional_token_bytes(token_bytes: Option<&[u8]>) -> String {
    token_bytes
        .map(format_token_bytes)
        .unwrap_or_else(|| "<no vocabulary bytes>".to_owned())
}

fn assert_mask_commit_equivalence(
    token_id: u32,
    token_bytes: Option<&[u8]>,
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
        format_optional_token_bytes(token_bytes),
        was_in_mask,
        commit_succeeded,
    );
}

pub(super) fn advance_special_token_paths(
    constraint: &Constraint,
    state: &BTreeMap<u32, ParserGSS>,
    token_id: u32,
) -> Option<ParserGSS> {
    let initial_state = constraint.tokenizer.initial_state();
    let initial_gss = state.get(&initial_state)?;
    let mut merged = None::<ParserGSS>;

    for special in constraint
        .special_token_terminals
        .iter()
        .filter(|special| special.token_id == token_id)
    {
        let pruned = prune_single_initial_state_for_terminal(
            initial_gss.clone(),
            initial_state,
            special.terminal_id,
            None,
        );
        if pruned.is_empty()
            || !stack_may_advance_on(&constraint.table, &pruned, special.terminal_id)
        {
            continue;
        }
        let advanced = advance_parser_stacks(constraint, &pruned, special.terminal_id);
        if advanced.is_empty() {
            continue;
        }
        merged = Some(match merged.take() {
            Some(existing) => existing.merge(&advanced),
            None => advanced,
        });
    }

    merged
}

#[derive(Default)]
struct SpecialTokenAdvanceProfile {
    paths: Option<ParserGSS>,
    prune_ns: u64,
    may_check_ns: u64,
    advance_ns: u64,
    summary_ns: u64,
    advances: Vec<AdvanceProfile>,
}

fn advance_special_token_paths_profiled(
    constraint: &Constraint,
    state: &BTreeMap<u32, ParserGSS>,
    token_id: u32,
    mut per_advance: Option<&mut Vec<PerAdvanceEntry>>,
) -> SpecialTokenAdvanceProfile {
    use std::time::Instant;

    let initial_state = constraint.tokenizer.initial_state();
    let Some(initial_gss) = state.get(&initial_state) else {
        return SpecialTokenAdvanceProfile::default();
    };
    let mut result = SpecialTokenAdvanceProfile::default();

    for special in constraint
        .special_token_terminals
        .iter()
        .filter(|special| special.token_id == token_id)
    {
        let prune_started_at = Instant::now();
        let pruned = prune_single_initial_state_for_terminal(
            initial_gss.clone(),
            initial_state,
            special.terminal_id,
            None,
        );
        result.prune_ns += prune_started_at.elapsed().as_nanos() as u64;
        if pruned.is_empty() {
            continue;
        }

        let may_started_at = Instant::now();
        let may_advance = stack_may_advance_on(&constraint.table, &pruned, special.terminal_id);
        result.may_check_ns += may_started_at.elapsed().as_nanos() as u64;
        if !may_advance {
            continue;
        }

        let advance_started_at = Instant::now();
        let (advanced, advance_profile) =
            advance_parser_stacks_profiled(constraint, &pruned, special.terminal_id);
        result.advance_ns += advance_started_at.elapsed().as_nanos() as u64;
        if advanced.is_empty() {
            continue;
        }

        if let Some(entries) = per_advance.as_deref_mut() {
            result.summary_ns += record_per_advance_entry(
                entries,
                initial_state,
                special.terminal_id,
                &pruned,
                &advanced,
                0,
                0,
                0,
                &[],
                advance_profile.clone(),
            );
        }
        result.advances.push(advance_profile);
        result.paths = Some(match result.paths.take() {
            Some(existing) => existing.merge(&advanced),
            None => advanced,
        });
    }

    result
}

fn apply_special_token_advance_profile(
    profile: &mut CommitProfile,
    special: &SpecialTokenAdvanceProfile,
) {
    profile.prune_ns += special.prune_ns;
    profile.advance_may_check_ns += special.may_check_ns;
    profile.may_advance_ns += special.may_check_ns;
    profile.advance_core_ns += special.advance_ns;
    profile.advance_ns += special.advance_ns;
    profile.adv_summary_ns += special.summary_ns;
    profile.n_advances += special.advances.len() as u64;
    for advance in &special.advances {
        apply_advance_profile(profile, advance);
    }
}

fn merge_special_token_paths(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    special_paths: Option<ParserGSS>,
) {
    let Some(gss) = special_paths.filter(|gss| !gss.is_empty()) else {
        return;
    };
    let initial_state = constraint.tokenizer.initial_state();
    state
        .entry(initial_state)
        .and_modify(|existing| *existing = existing.merge(&gss))
        .or_insert(gss);
}

fn finish_token_commit(state: &BTreeMap<u32, ParserGSS>) -> Result<(), String> {
    if state.is_empty() {
        Err("commit rejected: no valid parser states remain".to_owned())
    } else {
        Ok(())
    }
}

fn commit_token_impl(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    buffers: &mut CommitBuffers,
    token_id: u32,
) -> Result<(), String> {
    let bytes = token_bytes_for_id(constraint, token_id);
    let has_special = constraint.has_special_token_id(token_id);
    if bytes.is_none() && !has_special {
        return Err(format!(
            "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
        ));
    }

    let special_paths = has_special
        .then(|| advance_special_token_paths(constraint, state, token_id))
        .flatten();
    if let Some(bytes) = bytes {
        if commit_bytes_impl(constraint, state, bytes, buffers).is_err() {
            state.clear();
            buffers.clear_all();
        }
    } else {
        state.clear();
    }
    merge_special_token_paths(constraint, state, special_paths);
    finish_token_commit(state)
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

    fn intersects(&self, constraint: &Constraint, terminals: &BitSet) -> bool {
        match self {
            Self::SingleState(state_id) => {
                constraint.table.advance_row_intersects(*state_id, terminals)
            }
            Self::ManyStates(states) => states
                .iter()
                .any(|state_id| constraint.table.advance_row_intersects(*state_id, terminals)),
        }
    }
}

impl InitialCommitScan {
    fn collect(
        constraint: &Constraint,
        state: &BTreeMap<u32, ParserGSS>,
        bytes: &[u8],
    ) -> Self {
        let mut exec_results = FxHashMap::default();

        for &tokenizer_state in state.keys() {
            let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
            exec_results.insert(tokenizer_state, exec_result);
        }

        Self { exec_results }
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

fn has_actionable_terminal(
    actionable_terminals: Option<&ActionableTerminals>,
    constraint: &Constraint,
    terminals: &BitSet,
) -> bool {
    actionable_terminals
        .map(|actionable| actionable.intersects(constraint, terminals))
        .unwrap_or_else(|| !terminals.is_empty())
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

    if accepted_terminals.is_empty() && exec_result.end_state.is_empty() {
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
        for &end_state in &exec_result.end_state {
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

    if exec_result.end_state.is_empty() {
        return gss;
    }
    let relevant: TokenizerStateSet = exec_result
        .end_state
        .iter()
        .copied()
        .filter(|&end_state| {
            constraint
                .tokenizer
                .possible_future_terminals(end_state)
                .contains(terminal as usize)
        })
        .collect();
    if relevant.is_empty() {
        return gss;
    }

    gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
        let mut updated = terminals_disallowed.clone();
        for &end_state in &relevant {
            updated = updated.with_insert(end_state, terminal);
        }
        updated
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

/// Advance a deterministic lexer continuation without materializing a
/// `TokenizerExecResult`. Completed terminals are checked directly against the
/// parser action row. When none are ignored or actionable, the only semantic
/// effect is transporting token-start exclusions to the sole lexer end state.
fn commit_bytes_direct_continuation_fast_path(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Option<Result<(), String>> {
    if bytes.len() < 4 || constraint.tokenizer.has_epsilon_transitions() || state.len() != 1 {
        return None;
    }

    let (&start_state, gss) = state.iter().next().unwrap();
    let actionable_terminals = ActionableTerminals::from_gss(constraint, gss);
    let mut end_state = start_state;

    for &byte in bytes {
        end_state = constraint
            .tokenizer_fast_transitions
            .get(end_state as usize)
            .map_or(u32::MAX, |transitions| transitions[byte as usize]);
        if end_state == u32::MAX {
            return Some(Err(
                "commit rejected: no valid parser states remain".to_string(),
            ));
        }

        let matched_terminals = constraint.tokenizer.matched_terminal_bitset(end_state);
        let matched_ignore = constraint
            .ignore_terminal
            .is_some_and(|terminal| matched_terminals.contains(terminal as usize));
        let matched_actionable = has_actionable_terminal(
            actionable_terminals.as_ref(),
            constraint,
            matched_terminals,
        );
        if matched_ignore || matched_actionable {
            return None;
        }
    }

    if !end_state_may_advance(constraint, gss, end_state) {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }

    let accumulators_empty = gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());
    let (_, gss) = state.pop_first().unwrap();
    let transported = if accumulators_empty {
        gss
    } else {
        gss.apply_and_prune_no_promote(|td: &TerminalsDisallowed| {
            if td.is_empty() {
                return Some(TerminalsDisallowed::new());
            }
            let mut remapped = BTreeMap::new();
            if let Some(disallowed) = td.get(&start_state) {
                remapped.insert(end_state, disallowed.clone());
            }
            Some(TerminalsDisallowed(std::sync::Arc::new(remapped)))
        })
    };

    let fused = transported.fuse(Some(1));
    if fused.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }
    state.insert(end_state, fused);
    Some(Ok(()))
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

    let no_end_state = exec_result.end_state.is_empty();
    let accs_empty = gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());
    // The stale-exclusion bug that originally required the epsilon guard only
    // exists when exclusions must be transported from one NFA configuration
    // to another.  With empty accumulators this routine is already fully
    // state-set aware: it advances the sole full-width terminal and preserves
    // every viable lexer continuation independently.
    if constraint.tokenizer.has_epsilon_transitions() && !accs_empty {
        return None;
    }
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
            for &end_state in &exec_result.end_state {
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

    let end_states_to_keep: TokenizerStateSet = exec_result
        .end_state
        .iter()
        .copied()
        .filter(|&end_state| end_state_may_advance(constraint, &pruned_gss, end_state))
        .collect();

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
        advance_parser_stacks_owned(constraint, pruned_gss.clone(), terminal)
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

    if !end_states_to_keep.is_empty() {
        let fused = pruned_gss.fuse(Some(1));
        if !fused.is_empty() {
            for &end_state in &end_states_to_keep {
                state
                    .entry(end_state)
                    .and_modify(|existing| *existing = existing.merge(&fused))
                    .or_insert_with(|| fused.clone());
            }
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
    if constraint.tokenizer.has_epsilon_transitions()
        && state_has_nonempty_accumulators(state)
    {
        return None;
    }
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

        for &end_state in &exec_result.end_state {
            if end_state_may_advance(constraint, &pruned_gss, end_state) {
                merge_parser_state(&mut output, end_state, pruned_gss.clone());
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
    if constraint.tokenizer.has_epsilon_transitions() {
        return None;
    }
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

                  let advanced = if !template_advance_enabled()
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

            for &end_state in &exec_result.end_state {
                if end_state_may_advance(constraint, &gss_at_offset, end_state) {
                    merge_parser_state(&mut pending_state, end_state, gss_at_offset.clone());
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

        let finalizers = constraint
            .tokenizer_fast_finalizers
            .get(tokenizer_state as usize)
            .map_or(&[][..], |finalizers| finalizers.as_ref());
        for &terminal in finalizers {
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
                        .possible_future_terminals(end_state)
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
                            .possible_future_terminals(end_state)
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
                end_state: step.end_state.into_iter().collect(),
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
    if constraint.tokenizer.has_epsilon_transitions()
        && !gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
    {
        profile.failed_fast_path_probe_ns += total_start.elapsed().as_nanos() as u64;
        return None;
    }

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

    let no_end_state = exec_result.end_state.is_empty();
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
            for &end_state in &exec_result.end_state {
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
    let end_states_to_keep: TokenizerStateSet = exec_result
        .end_state
        .iter()
        .copied()
        .filter(|&end_state| end_state_may_advance(constraint, &pruned_gss, end_state))
        .collect();
    profile.fast_path_end_state_check_ns = end_state_check_start.elapsed().as_nanos() as u64;

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

    if !end_states_to_keep.is_empty() {
        let fuse_start = Instant::now();
        let fused = pruned_gss.fuse(Some(1));
        let fuse_elapsed = fuse_start.elapsed().as_nanos() as u64;
        profile.fast_path_fuse_ns += fuse_elapsed;
        profile.fuse_ns += fuse_elapsed;
        if !fused.is_empty() {
            let update_start = Instant::now();
            for &end_state in &end_states_to_keep {
                state
                    .entry(end_state)
                    .and_modify(|existing| *existing = existing.merge(&fused))
                    .or_insert_with(|| fused.clone());
            }
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

    if allow_fast_paths
        && constraint.tokenizer.has_epsilon_transitions()
        && state.len() == 1
    {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_start = Instant::now();
        let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
        let exec_elapsed = exec_start.elapsed().as_nanos() as u64;
        profile.initial_exec_ns = exec_elapsed;
        profile.exec_ns = exec_elapsed;
        profile.fast_path_tokenizer_exec_ns = exec_elapsed;
        if let Some(result) = commit_bytes_fast_path_profiled(
            constraint,
            state,
            bytes,
            tokenizer_state,
            &exec_result,
            advances.as_deref_mut(),
            &mut profile,
        ) {
            return result.map(|()| profile);
        }
    }

    if allow_fast_paths && !constraint.tokenizer.has_epsilon_transitions() && state.len() == 1 {
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
            let linear_fast_path_eligible = !exec_result.end_state.iter().copied().any(|end_state| {
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

                                for &end_state in &exec_result.end_state {
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
                                        gss_at_offset.clone(),
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
        for (tokenizer_state, mut gss_at_offset) in states_to_process {
            profile.n_queue_entries += 1;

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

            if offset == 0
                && !gss_at_offset
                    .all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
            {
                let prune_start = Instant::now();
                gss_at_offset = prune_single_initial_state_for_exec(
                    constraint,
                    gss_at_offset,
                    tokenizer_state,
                    &exec_result,
                );
                profile.prune_ns += prune_start.elapsed().as_nanos() as u64;
                if gss_at_offset.is_empty() {
                    continue;
                }
            }

            let actionable_start = Instant::now();
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            profile.actionable_ns += actionable_start.elapsed().as_nanos() as u64;

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

            for &end_state in &exec_result.end_state {
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
                    gss_at_offset.clone(),
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

fn clear_state_on_commit_error<T>(
    state: &mut BTreeMap<u32, ParserGSS>,
    result: Result<T, String>,
) -> Result<T, String> {
    if result.is_err() {
        state.clear();
    }
    result
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

        if exec_result.end_state.len() > 1 {
            return if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
        }
        if let Some(end_state) = exec_result.end_state.first().copied()
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
                        .possible_future_terminals(end_state)
                        .contains(terminal as usize)
            });
            if !keep_carried {
                gss = carried_stack.take().unwrap().into_gss();
            }
        }

        if let Some(end_state) = exec_result.end_state.first().copied() {
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
                && exec_result.end_state.iter().copied().all(|end_state| {
                    end_state != constraint.tokenizer.initial_state()
                        && !constraint.table.advance_row_intersects(
                            top_state,
                            constraint.tokenizer.possible_future_terminals(end_state),
                        )
                        && !constraint
                            .tokenizer
                            .possible_future_terminals(end_state)
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
        if exec_result.end_state.len() > 1 {
            profile.linear_fast_path_end_state_check_ns +=
                end_state_start.elapsed().as_nanos() as u64;
            profile.linear_fast_path_total_ns = total_start.elapsed().as_nanos() as u64;
            return if offset > 0 {
                LinearFastPathResult::Continue { gss, offset }
            } else {
                LinearFastPathResult::Restart
            };
        }
        if let Some(end_state) = exec_result.end_state.first().copied() {
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

    if let Some(result) = commit_bytes_direct_continuation_fast_path(constraint, state, bytes) {
        return result;
    }

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

    if !constraint.tokenizer.has_epsilon_transitions() && state.len() == 1 {
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
            .iter()
            .copied()
            .any(|end_state| {
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

                            for &end_state in &exec_result.end_state {
                                if !end_state_may_advance(constraint, &gss_at_offset, end_state) {
                                    continue;
                                }

                                queue_parser_state(
                                    &mut processing_queue,
                                    &mut bufs.pending_state,
                                    bytes.len(),
                                    bytes.len(),
                                    end_state,
                                    gss_at_offset.clone(),
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
        bufs.exec_results.insert(tokenizer_state, exec_result);
    } else {
        bufs.clear_all();

        for &tokenizer_state in state.keys() {
            let exec_result = execute_tokenizer_from_state_small(constraint, bytes, tokenizer_state);
            bufs.exec_results.insert(tokenizer_state, exec_result);
        }
    }

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
        for (tokenizer_state, mut gss_at_offset) in states_to_process {
            let exec_result = if offset == 0 {
                bufs.exec_results.remove(&tokenizer_state).unwrap_or_else(|| {
                    execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
                })
            } else {
                execute_tokenizer_from_state_small(constraint, &bytes[offset..], tokenizer_state)
            };

            if offset == 0
                && !gss_at_offset
                    .all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty())
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

            for &end_state in &exec_result.end_state {
                if !end_state_may_advance(constraint, &gss_at_offset, end_state) {
                    continue;
                }

                queue_parser_state(
                    &mut processing_queue,
                    &mut bufs.pending_state,
                    bytes.len(),
                    bytes.len(),
                    end_state,
                    gss_at_offset.clone(),
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
    /// `token_id` must either exist in the vocabulary the constraint was built
    /// with or be declared by a special-token terminal in the grammar.
    /// Committing a token that is grammatically invalid (not in the current
    /// mask) drives the constraint into a fail state — this is normal and
    /// observable via an all-zero mask.
    ///
    /// # Errors
    ///
    /// Returns an error if `token_id` is neither present in the vocabulary nor
    /// declared by a special-token terminal.
    pub fn commit_token(
        &mut self,
        token_id: u32,
    ) -> Result<(), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let result = commit_token_impl(constraint, &mut self.state, &mut self.buffers, token_id);
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    pub(crate) fn commit_token_dynamic(&mut self, token_id: u32) -> Result<(), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let was_in_mask = if commit_mask_assert_enabled() {
            let mut mask = vec![0u32; constraint.mask_len()];
            self.fill_mask_dynamic(&mut mask);
            Some(token_in_mask(&mask, token_id))
        } else {
            None
        };
        let result = commit_token_impl(constraint, &mut self.state, &mut self.buffers, token_id);
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    pub(crate) fn commit_tokens_dynamic(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token_dynamic(token)?;
        }
        Ok(())
    }

    pub(crate) fn commit_token_timed_ns(&mut self, token_id: u32) -> Result<u64, String> {
        use std::time::Instant;

        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let start = Instant::now();
        let result = commit_token_impl(constraint, &mut self.state, &mut self.buffers, token_id);
        let total_ns = start.elapsed().as_nanos() as u64;
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result.map(|()| total_ns)
    }

    pub(crate) fn commit_token_profiled(&mut self, token_id: u32) -> Result<CommitProfile, String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let has_special = constraint.has_special_token_id(token_id);
        if bytes.is_none() && !has_special {
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let total_started_at = std::time::Instant::now();
        let special = if has_special {
            advance_special_token_paths_profiled(constraint, &self.state, token_id, None)
        } else {
            SpecialTokenAdvanceProfile::default()
        };
        let mut profile = if let Some(bytes) = bytes {
            match commit_bytes_impl_profiled(
                constraint,
                &mut self.state,
                bytes,
                &mut self.buffers,
                None,
                true,
            ) {
                Ok(profile) => profile,
                Err(_) => {
                    self.state.clear();
                    self.buffers.clear_all();
                    CommitProfile::default()
                }
            }
        } else {
            self.state.clear();
            CommitProfile::default()
        };
        apply_special_token_advance_profile(&mut profile, &special);
        merge_special_token_paths(constraint, &mut self.state, special.paths);
        profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
        let result = finish_token_commit(&self.state).map(|()| profile);
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    pub(crate) fn commit_token_per_advance(
        &mut self,
        token_id: u32,
    ) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id);
        let has_special = constraint.has_special_token_id(token_id);
        if bytes.is_none() && !has_special {
            return Err(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            ));
        }
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let total_started_at = std::time::Instant::now();
        let mut advances = Vec::new();
        let special = if has_special {
            advance_special_token_paths_profiled(
                constraint,
                &self.state,
                token_id,
                Some(&mut advances),
            )
        } else {
            SpecialTokenAdvanceProfile::default()
        };
        let mut profile = if let Some(bytes) = bytes {
            match commit_bytes_impl_profiled(
                constraint,
                &mut self.state,
                bytes,
                &mut self.buffers,
                Some(&mut advances),
                profile_allow_fast_paths(),
            ) {
                Ok(profile) => profile,
                Err(_) => {
                    self.state.clear();
                    self.buffers.clear_all();
                    advances.clear();
                    CommitProfile::default()
                }
            }
        } else {
            self.state.clear();
            CommitProfile::default()
        };
        apply_special_token_advance_profile(&mut profile, &special);
        merge_special_token_paths(constraint, &mut self.state, special.paths);
        profile.total_ns = total_started_at.elapsed().as_nanos() as u64;
        let result = finish_token_commit(&self.state)
            .map(|()| (advances, final_stacks(&self.state), profile));
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let result = commit_bytes_impl(self.constraint, &mut self.state, bytes, &mut self.buffers);
        let result = clear_state_on_commit_error(&mut self.state, result);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraint, Vocab};

    type CanonicalCommitState =
        Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)>;

    fn canonical_commit_state(
        state: &BTreeMap<u32, ParserGSS>,
    ) -> CanonicalCommitState {
        state
            .iter()
            .map(|(&tokenizer_state, gss)| {
                let mut stacks = gss
                    .to_stacks()
                    .into_iter()
                    .map(|(stack, terminals_disallowed)| {
                        let disallowed = terminals_disallowed
                            .iter()
                            .map(|(&state, terminals)| {
                                (state, terminals.iter().copied().collect::<Vec<_>>())
                            })
                            .collect::<Vec<_>>();
                        (stack, disallowed)
                    })
                    .collect::<Vec<_>>();
                stacks.sort();
                (tokenizer_state, stacks)
            })
            .collect()
    }

    #[test]
    fn direct_continuation_fast_path_ignores_non_actionable_matches_exactly() {
        let vocab = Vocab::new(vec![(0, b"abcd".to_vec())], None);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "abcdef";
                t B ::= "a";
                t X ::= "x";
                nt start ::= A | X B;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(!constraint.tokenizer.has_epsilon_transitions());

        let bytes = vocab.entries.get(&0).unwrap();
        let mut fast = constraint.start();
        let mut general = constraint.start();
        let exec_result = execute_tokenizer_from_state_small(
            &constraint,
            bytes,
            *fast.state.keys().next().unwrap(),
        );
        assert!(
            exec_result.matches.iter().any(|matched| matched.width == 1),
            "test requires a completed non-actionable prefix match: {exec_result:?}",
        );

        let fast_result =
            commit_bytes_direct_continuation_fast_path(&constraint, &mut fast.state, bytes)
                .expect("direct continuation fast path should apply");
        let general_result = commit_bytes_impl_profiled(
            &constraint,
            &mut general.state,
            bytes,
            &mut general.buffers,
            None,
            false,
        );

        assert_eq!(fast_result.is_ok(), general_result.is_ok());
        assert_eq!(fast.state, general.state);
    }

    #[test]
    fn direct_continuation_fast_path_transports_nonempty_accumulators_exactly() {
        let vocab = Vocab::new(vec![(0, b"abcd".to_vec())], None);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "abcdef";
                t B ::= "a";
                t X ::= "x";
                nt start ::= A | X B;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(!constraint.tokenizer.has_epsilon_transitions());

        let bytes = vocab.entries.get(&0).unwrap();
        let mut fast = constraint.start();
        let tokenizer_state = *fast.state.keys().next().unwrap();
        let decorated = fast
            .state
            .get(&tokenizer_state)
            .unwrap()
            .apply(|td: &TerminalsDisallowed| td.with_insert(tokenizer_state, 12345));
        fast.state.insert(tokenizer_state, decorated.clone());
        let mut general = fast.clone();

        let fast_result =
            commit_bytes_direct_continuation_fast_path(&constraint, &mut fast.state, bytes)
                .expect("direct continuation fast path should transport exclusions");
        let general_result = commit_bytes_impl_profiled(
            &constraint,
            &mut general.state,
            bytes,
            &mut general.buffers,
            None,
            false,
        );

        assert_eq!(fast_result.is_ok(), general_result.is_ok());
        assert_eq!(
            canonical_commit_state(&fast.state),
            canonical_commit_state(&general.state),
        );
    }

    #[test]
    fn direct_continuation_fast_path_defers_actionable_matches() {
        let vocab = Vocab::new(vec![(0, b"abcd".to_vec())], None);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "abcd";
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let mut state = constraint.start();
        assert!(
            commit_bytes_direct_continuation_fast_path(
                &constraint,
                &mut state.state,
                vocab.entries.get(&0).unwrap(),
            )
            .is_none(),
            "actionable terminal completion must fall through",
        );
    }

    #[test]
    fn rejected_public_commits_enter_fail_state() {
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec())],
            None,
        );
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();

        let assert_failed = |state: &ConstraintState<'_>| {
            assert!(state.state.is_empty());
            assert!(state.mask().iter().all(|&word| word == 0));
        };

        let mut state = constraint.start();
        assert!(state.commit_token(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_timed_ns(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_profiled(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_token_per_advance(1).is_err());
        assert_failed(&state);

        let mut state = constraint.start();
        assert!(state.commit_bytes(b"b").is_err());
        assert_failed(&state);
    }

    fn assert_fast_and_general_queue_match<'a>(
        constraint: &'a Constraint,
        fast_state: &ConstraintState<'a>,
        token_id: u32,
        bytes: &[u8],
        context: &str,
    ) -> Option<ConstraintState<'a>> {
        let mut fast = fast_state.clone();
        let mut profiled = fast_state.clone();
        let mut general = fast_state.clone();

        let fast_result = commit_bytes_impl(
            constraint,
            &mut fast.state,
            bytes,
            &mut fast.buffers,
        );
        let profiled_result = commit_bytes_impl_profiled(
            constraint,
            &mut profiled.state,
            bytes,
            &mut profiled.buffers,
            None,
            true,
        );
        let general_result = commit_bytes_impl_profiled(
            constraint,
            &mut general.state,
            bytes,
            &mut general.buffers,
            None,
            false,
        );

        assert_eq!(
            fast_result.is_ok(),
            general_result.is_ok(),
            "commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast={:?}\ngeneral={:?}",
            fast.state,
            general.state,
        );
        assert_eq!(
            profiled_result.is_ok(),
            general_result.is_ok(),
            "profiled commit result mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled={:?}\ngeneral={:?}",
            profiled.state,
            general.state,
        );
        if fast_result.is_err() {
            return None;
        }
        assert_eq!(
            canonical_commit_state(&fast.state),
            canonical_commit_state(&general.state),
            "successful commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nfast_stacks={:#?}\ngeneral_stacks={:#?}",
            fast.state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks()))
                .collect::<Vec<_>>(),
            general
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks()))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            canonical_commit_state(&profiled.state),
            canonical_commit_state(&general.state),
            "successful profiled commit state mismatch: {context} token_id={token_id} bytes={bytes:?}\nprofiled_stacks={:#?}\ngeneral_stacks={:#?}",
            profiled
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks()))
                .collect::<Vec<_>>(),
            general
                .state
                .iter()
                .map(|(&ts, gss)| (ts, gss.to_stacks()))
                .collect::<Vec<_>>(),
        );

        Some(fast)
    }

    #[test]
    fn monolithic_commit_fast_paths_match_general_queue_on_small_language_space() {
        const WORDS: [&str; 4] = ["a", "b", "ab", "ba"];
        let vocab = Vocab::new(
            WORDS
                .iter()
                .enumerate()
                .map(|(id, word)| (id as u32, word.as_bytes().to_vec()))
                .collect(),
            None,
        );
        let languages = (1u32..1u32 << WORDS.len())
            .filter(|mask| mask.count_ones() <= 2)
            .collect::<Vec<_>>();

        let rule = |name: &str, mask: u32| {
            let rhs = WORDS
                .iter()
                .enumerate()
                .filter_map(|(index, word)| {
                    (mask & (1 << index) != 0).then(|| format!("\"{word}\""))
                })
                .collect::<Vec<_>>()
                .join(" | ");
            format!("t {name} ::= {rhs};\n")
        };

        for &a in &languages {
            for &b in &languages {
                let grammar = format!(
                    "start start;\n{}{}nt item ::= A | B;\nnt start ::= item item? item?;\n",
                    rule("A", a),
                    rule("B", b),
                );
                let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
                let mut frontier = vec![(constraint.start(), Vec::<u32>::new())];

                for depth in 0..3 {
                    let mut next = Vec::new();
                    for (state, path) in frontier {
                        let mask = state.mask();
                        for (&token_id, bytes) in vocab.entries.iter() {
                            let context = format!(
                                "A_mask={a:#06b} B_mask={b:#06b} depth={depth} path={path:?}\ngrammar:\n{grammar}"
                            );
                            let next_state = assert_fast_and_general_queue_match(
                                &constraint,
                                &state,
                                token_id,
                                bytes,
                                &context,
                            );
                            let token_in_mask = mask
                                .get(token_id as usize / 32)
                                .is_some_and(|word| {
                                    word & (1u32 << (token_id % 32)) != 0
                                });
                            assert_eq!(
                                token_in_mask,
                                next_state.is_some(),
                                "mask/commit mismatch: {context} token_id={token_id} bytes={bytes:?}"
                            );
                            if let Some(next_state) = next_state {
                                let mut next_path = path.clone();
                                next_path.push(token_id);
                                next.push((next_state, next_path));
                            }
                        }
                    }
                    frontier = next;
                }
            }
        }
    }

    #[test]
    fn residual_bc_fast_path_matches_general_queue() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"ab".to_vec()),
                (4, b"ba".to_vec()),
                (5, b"bc".to_vec()),
                (6, b"abc".to_vec()),
            ],
            None,
        );
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a" | "ab";
                t B ::= "bc";
                nt item ::= A | B;
                nt start ::= item item? item?;
            "#,
            &vocab,
        )
        .unwrap();

        let mut fast = constraint.start();
        let mut slow = constraint.start();
        let fast_result = commit_bytes_impl(
            &constraint,
            &mut fast.state,
            vocab.entries.get(&0).unwrap(),
            &mut fast.buffers,
        );
        fast.generation += 1;
        assert!(fast_result.is_ok());
        let slow_result = commit_bytes_impl_profiled(
            &constraint,
            &mut slow.state,
            vocab.entries.get(&0).unwrap(),
            &mut slow.buffers,
            None,
            false,
        );
        slow.generation += 1;
        assert!(slow_result.is_ok());
        assert_eq!(fast.state, slow.state, "state mismatch after token a");

        let mut next_fast = fast.clone();
        let mut next_slow = slow.clone();
        let fast_result = commit_bytes_impl(
            &constraint,
            &mut next_fast.state,
            vocab.entries.get(&5).unwrap(),
            &mut next_fast.buffers,
        );
        let slow_result = commit_bytes_impl_profiled(
            &constraint,
            &mut next_slow.state,
            vocab.entries.get(&5).unwrap(),
            &mut next_slow.buffers,
            None,
            false,
        );
        assert_eq!(
            fast_result.is_ok(),
            slow_result.is_ok(),
            "fast={:?}\nslow={:?}",
            next_fast.state,
            next_slow.state,
        );
        assert_eq!(next_fast.state, next_slow.state);
    }

    #[test]
    fn epsilon_commit_fast_paths_match_no_fast_path_reference() {
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"c".to_vec()),
                (3, b"aa".to_vec()),
                (4, b"ab".to_vec()),
                (5, b" ".to_vec()),
                (6, b" a".to_vec()),
                (7, b"a ".to_vec()),
                (8, b" a ".to_vec()),
                (9, b"abc".to_vec()),
                (10, b"aab".to_vec()),
            ],
            None,
        );
        let grammar = crate::grammar::glrm::from_glrm(
            r#"
                start start;
                ignore WS;
                lexer group ws ::= WS;
                lexer group a ::= A;
                lexer group b ::= B;
                lexer group c ::= C;
                t WS ::= " "+;
                t A ::= "a"+;
                t B ::= "b";
                t C ::= "c";
                nt item ::= A | B | C;
                nt start ::= item item? item?;
            "#,
        )
        .unwrap();
        let grammar = crate::grammar::ast::lower(&grammar).unwrap();
        let constraint = crate::compiler::pipeline::compile_owned_with_lexer_adaptive(
            grammar,
            &vocab,
            false,
        );
        assert!(constraint.tokenizer.has_epsilon_transitions());

        let mut frontier = vec![(
            constraint.start(),
            constraint.start(),
            constraint.start(),
            Vec::<u32>::new(),
        )];
        for depth in 0..=4 {
            let mut next = Vec::new();
            for (fast, profiled, general, path) in frontier {
                assert_eq!(
                    fast.mask(),
                    general.mask(),
                    "epsilon mask mismatch after path {path:?}\nfast={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&fast.state),
                    canonical_commit_state(&general.state),
                );
                assert_eq!(
                    profiled.mask(),
                    general.mask(),
                    "epsilon profiled-mask mismatch after path {path:?}\nprofiled={:#?}\ngeneral={:#?}",
                    canonical_commit_state(&profiled.state),
                    canonical_commit_state(&general.state),
                );
                assert_eq!(
                    fast.is_finished(),
                    general.is_finished(),
                    "epsilon completion mismatch after path {path:?}",
                );
                assert_eq!(
                    profiled.is_finished(),
                    general.is_finished(),
                    "epsilon profiled completion mismatch after path {path:?}",
                );
                if depth == 4 {
                    continue;
                }

                for (&token_id, bytes) in vocab.entries.iter() {
                    let mut next_fast = fast.clone();
                    let mut next_profiled = profiled.clone();
                    let mut next_general = general.clone();
                    let fast_result = commit_bytes_impl(
                        &constraint,
                        &mut next_fast.state,
                        bytes,
                        &mut next_fast.buffers,
                    );
                    let profiled_result = commit_bytes_impl_profiled(
                        &constraint,
                        &mut next_profiled.state,
                        bytes,
                        &mut next_profiled.buffers,
                        None,
                        true,
                    );
                    let general_result = commit_bytes_impl_profiled(
                        &constraint,
                        &mut next_general.state,
                        bytes,
                        &mut next_general.buffers,
                        None,
                        false,
                    );
                    assert_eq!(
                        fast_result.is_ok(),
                        general_result.is_ok(),
                        "epsilon commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nfast={:#?}\ngeneral={:#?}",
                        canonical_commit_state(&next_fast.state),
                        canonical_commit_state(&next_general.state),
                    );
                    assert_eq!(
                        profiled_result.is_ok(),
                        general_result.is_ok(),
                        "epsilon profiled commit result mismatch after path {path:?} token_id={token_id} bytes={bytes:?}\nprofiled={:#?}\ngeneral={:#?}",
                        canonical_commit_state(&next_profiled.state),
                        canonical_commit_state(&next_general.state),
                    );
                    if fast_result.is_ok() {
                        let mut next_path = path.clone();
                        next_path.push(token_id);
                        next.push((next_fast, next_profiled, next_general, next_path));
                    }
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn epsilon_full_width_terminal_with_empty_accumulators_uses_fast_path() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);
        let grammar = crate::grammar::glrm::from_glrm(
            r#"
                start start;
                lexer group left ::= A;
                lexer group right ::= B;
                t A ::= "a";
                t B ::= "b";
                nt start ::= A | B;
            "#,
        )
        .unwrap();
        let grammar = crate::grammar::ast::lower(&grammar).unwrap();
        let constraint = crate::compiler::pipeline::compile_owned_with_lexer_adaptive(
            grammar,
            &vocab,
            false,
        );
        assert!(constraint.tokenizer.has_deterministic_dispatch());

        let mut state = constraint.start();
        let profile = state.commit_token_profiled(0).unwrap();
        assert!(profile.fast_path_total_ns > 0, "profile={profile:?}");
        assert_eq!(profile.n_queue_entries, 0, "profile={profile:?}");
        assert_eq!(profile.n_advances, 1, "profile={profile:?}");
    }
}
