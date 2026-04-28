use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::automata::lexer::tokenizer::TokenizerExecResult;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{
    ParserGSS,
    advance_stacks,
    advance_stacks_owned,
    advance_stacks_profiled,
    AdvanceProfile,
    stack_may_advance_on,
    stack_may_advance_on_any,
};
use crate::compiler::glr::table::Action;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::{ConstraintState, CommitBuffers};
use rustc_hash::{FxHashMap, FxHashSet};

type ParserStatesByTokenizer = FxHashMap<u32, ParserGSS>;

/// Cache for `advance_stacks` results, keyed by (GSS pointer, terminal).
/// Stores the key GSS alongside the result to keep its Arc alive and prevent
/// address reuse (ABA problem) within a single `commit_bytes_impl` call.
type AdvanceResultCache = FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>;

struct InitialCommitScan {
    exec_results: FxHashMap<u32, TokenizerExecResult>,
    remapped_tokenizer_states: FxHashMap<u32, u32>,
    accepted_terminals: FxHashMap<u32, FxHashSet<u32>>,
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
        "commit/mask mismatch for token_id {} bytes {:?}: token_in_mask={} commit_succeeded={}",
        token_id,
        token_bytes,
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
    Many(FxHashSet<u32>),
}

impl ActionableTerminals {
    fn from_gss(constraint: &Constraint, gss: &ParserGSS) -> Option<Self> {
        if let Some(state_id) = gss.single_top_value() {
            return Some(Self::SingleState(state_id));
        }

        let mut terminals = FxHashSet::default();
        for state_id in gss.peek_values() {
            if let Some(by_terminal) = constraint.table.action.get(state_id as usize) {
                terminals.extend(by_terminal.keys().copied());
            }
        }

        if terminals.is_empty() {
            None
        } else {
            Some(Self::Many(terminals))
        }
    }

    fn contains(&self, constraint: &Constraint, terminal: u32) -> bool {
        match self {
            Self::SingleState(state_id) => constraint.table.action(*state_id, terminal).is_some(),
            Self::Many(terminals) => terminals.contains(&terminal),
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
            let exec_result = constraint.tokenizer.execute_from_state(bytes, tokenizer_state);

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

fn prune_initial_states(
    state: &mut BTreeMap<u32, ParserGSS>,
    accepted_terminals: &FxHashMap<u32, FxHashSet<u32>>,
    remapped_tokenizer_states: &FxHashMap<u32, u32>,
) {
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
        let advanced = advance_stacks(&constraint.table, gss_at_offset, terminal);
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

    // Check if end_state needs processing
    if let Some(end_state) = exec_result.end_state {
        if end_state_may_advance(constraint, gss, end_state) {
            return None;
        }
    }

    let no_end_state = exec_result.end_state.is_none();
    let all_accs_empty = no_end_state
        && gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty());

    // Ultra-fast path: single Interface, empty accs, no end_state, pure shift.
    // Inlines the entire advance + prune + fuse to avoid all function call overhead.
    if all_accs_empty {
        if let Some(top_state) = gss.single_exclusive_top_value() {
            if let Some(Action::Shift(target, is_replace)) = constraint.table.action(top_state, terminal) {
                let shifted = if *is_replace {
                    gss.popn(1).push(*target)
                } else {
                    gss.push(*target)
                };
                state.clear();
                state.insert(constraint.tokenizer.initial_state(), shifted);
                return Some(Ok(()));
            }
            if let Some(Action::StackShifts(shifts)) = constraint.table.action(top_state, terminal) {
                let mut shifted = ParserGSS::empty();
                for shift in shifts {
                    let mut branch = gss.popn(shift.pop as isize);
                    for &target in &shift.pushes {
                        branch = branch.push(target);
                    }
                    shifted = if shifted.is_empty() { branch } else { shifted.merge(&branch) };
                }
                state.clear();
                state.insert(constraint.tokenizer.initial_state(), shifted);
                return Some(Ok(()));
            }
        }
    }

    // Take ownership of the GSS for the standard fast path.
    // This allows advance_stacks_owned to avoid cloning the inner Arc.
    let (_, gss_owned) = state.pop_first().unwrap();

    // Standard fast path: skip prune when accumulators are empty.
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
            return Some(Err(
                "commit rejected: no valid parser states remain".to_string(),
            ));
        }
        pruned
    };

    // Advance the parser — use owned variant to avoid initial Arc clone
    let advanced = advance_stacks_owned(&constraint.table, pruned_gss, terminal);
    if advanced.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }

    let advanced =
        apply_future_terminal_disallow(constraint, &exec_result, terminal, advanced);
    let fused = advanced.fuse(Some(1));

    if fused.is_empty() {
        return Some(Err(
            "commit rejected: no valid parser states remain".to_string(),
        ));
    }

    state.insert(constraint.tokenizer.initial_state(), fused);
    Some(Ok(()))
}

enum LinearFastPathResult {
    Complete(Result<ParserGSS, String>),
    Continue { gss: ParserGSS, offset: usize },
    Restart,
}

fn commit_bytes_linear_fast_path(
    constraint: &Constraint,
    start_gss: ParserGSS,
    bytes: &[u8],
    first_exec_result: TokenizerExecResult,
) -> LinearFastPathResult {
    let ignore_terminal = constraint.ignore_terminal;
    let mut gss = start_gss;
    let mut offset = 0usize;
    let mut exec_result = first_exec_result;

    loop {
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
            if !stack_may_advance_on(&constraint.table, &gss, terminal) {
                return if offset > 0 {
                    LinearFastPathResult::Continue { gss, offset }
                } else {
                    LinearFastPathResult::Restart
                };
            }

            let advanced = advance_stacks_owned(&constraint.table, gss, terminal);
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
            let fused = gss.fuse(Some(1));
            if fused.is_empty() {
                return LinearFastPathResult::Complete(Err(
                    "commit rejected: no valid parser states remain".to_string(),
                ));
            }
            return LinearFastPathResult::Complete(Ok(fused));
        }

        exec_result = constraint
            .tokenizer
            .execute_from_state(&bytes[offset..], constraint.tokenizer.initial_state());
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

    // Single tokenizer state: execute tokenizer ONCE, try fast path, reuse result
    if state.len() == 1 {
        let (&tokenizer_state, _) = state.iter().next().unwrap();
        let exec_result = constraint.tokenizer.execute_from_state(bytes, tokenizer_state);

        // Try fast path with pre-computed exec_result
        if let Some(result) = commit_bytes_fast_path(
            constraint, state, bytes, tokenizer_state, &exec_result,
        ) {
            return result;
        }

        if state
            .values()
            .next()
            .is_some_and(|gss| gss.all_accs_satisfy(|td: &TerminalsDisallowed| td.is_empty()))
            && !exec_result
                .end_state
                .is_some_and(|end_state| {
                    state
                        .values()
                        .next()
                        .is_some_and(|gss| end_state_may_advance(constraint, gss, end_state))
                })
        {
            let start_gss = state.values().next().unwrap().clone();
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
                            let exec_result = constraint
                                .tokenizer
                                .execute_from_state(&bytes[offset..], tokenizer_state);

                            bufs.seen_matches.clear();
                            bufs.terminal_result_cache.clear();

                            for matched in &exec_result.matches {
                                let new_offset = offset + matched.width;
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
                                if !bufs.seen_matches.insert((matched.width, matched.id)) {
                                    continue;
                                }

                                if ignored {
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
                                    matched.id,
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
            let exec_result = constraint.tokenizer.execute_from_state(bytes, tokenizer_state);

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
                    constraint
                        .tokenizer
                        .execute_from_state(&bytes[offset..], tokenizer_state)
                })
            } else {
                constraint
                    .tokenizer
                    .execute_from_state(&bytes[offset..], tokenizer_state)
            };

            bufs.seen_matches.clear();
            bufs.terminal_result_cache.clear();

            for matched in &exec_result.matches {
                let new_offset = offset + matched.width;
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
                if !bufs.seen_matches.insert((matched.width, matched.id)) {
                    continue;
                }

                if ignored {
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
                    matched.id,
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

    Ok(())
}

pub(crate) fn token_bytes_may_commit_from_state_map(
    constraint: &Constraint,
    state: &BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
) -> bool {
    let mut probe_state = state.clone();
    bufs.clear_all();
    commit_bytes_impl(constraint, &mut probe_state, bytes, bufs).is_ok()
}

fn commit_bytes_impl_profiled(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Result<(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64), String> {
    use std::time::Instant;
    let verbose = std::env::var("GLRMASK_PROFILE_VERBOSE").is_ok();
    let t_total = Instant::now();

    if bytes.is_empty() {
        let total_ns = t_total.elapsed().as_nanos() as u64;
        return Ok((total_ns, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    }

    let ignore_terminal = constraint.ignore_terminal;
    let n_tokenizer_states = state.len() as u64;

    let t_scan = Instant::now();
    let mut initial_scan = InitialCommitScan::collect(constraint, state, bytes);
    let scan_ns = t_scan.elapsed().as_nanos() as u64;

    let t_prune = Instant::now();
    prune_initial_states(
        state,
        &initial_scan.accepted_terminals,
        &initial_scan.remapped_tokenizer_states,
    );
    state.retain(|_, parser_state| !parser_state.is_empty());
    let prune_ns = t_prune.elapsed().as_nanos() as u64;

    let t_queue = Instant::now();
    let t_queue_alloc = Instant::now();
    let mut pending_state = ParserStatesByTokenizer::default();
    let _advance_result_cache = AdvanceResultCache::default();
    let mut processing_queue: Vec<ParserStatesByTokenizer> =
        (0..=bytes.len()).map(|_| ParserStatesByTokenizer::default()).collect();
    processing_queue[0] = std::mem::take(state).into_iter().collect();
    let queue_alloc_ns = t_queue_alloc.elapsed().as_nanos() as u64;

    let mut n_queue_entries: u64 = 0;
    let mut exec_ns: u64 = 0;
    let mut advance_ns: u64 = 0;
    let advance_may_check_ns: u64 = 0;
    let mut advance_core_ns: u64 = 0;
    let mut advance_future_disallow_ns: u64 = 0;
    let mut actionable_ns: u64 = 0;
    let mut may_advance_ns: u64 = 0;
    let mut n_advances: u64 = 0;
    let mut queue_state_ns: u64 = 0;
    let mut seen_matches_ns: u64 = 0;
    let mut profile_agg_ns: u64 = 0;
    let mut n_matches_checked: u64 = 0;
    let mut n_matches_skipped_actionable: u64 = 0;
    let mut n_matches_skipped_seen: u64 = 0;
    let mut n_ignored: u64 = 0;
    let mut adv_profile = AdvanceProfile::default();
    let mut offset = 0usize;
    while offset < processing_queue.len() {
        if processing_queue[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut processing_queue[offset]);
        for (tokenizer_state, gss_at_offset) in states_to_process {
            n_queue_entries += 1;
            let t_act = Instant::now();
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            actionable_ns += t_act.elapsed().as_nanos() as u64;
            let t_exec = Instant::now();
            let exec_result = if offset == 0 {
                initial_scan.take_exec_result(tokenizer_state).unwrap_or_else(|| {
                    constraint.tokenizer.execute_from_state(&bytes[offset..], tokenizer_state)
                })
            } else {
                constraint.tokenizer.execute_from_state(&bytes[offset..], tokenizer_state)
            };
            exec_ns += t_exec.elapsed().as_nanos() as u64;

            let mut seen_matches = FxHashSet::default();
            let _terminal_result_cache = FxHashMap::<u32, ParserGSS>::default();

            for matched in &exec_result.matches {
                n_matches_checked += 1;
                let new_offset = offset + matched.width;
                let ignored = is_ignored_terminal(ignore_terminal, matched.id);

                if !ignored
                    && !is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id)
                {
                    n_matches_skipped_actionable += 1;
                    continue;
                }
                let t_seen = Instant::now();
                let is_new = seen_matches.insert((matched.width, matched.id));
                seen_matches_ns += t_seen.elapsed().as_nanos() as u64;
                if !is_new {
                    n_matches_skipped_seen += 1;
                    continue;
                }

                if ignored {
                    n_ignored += 1;
                    let t_qs = Instant::now();
                    queue_parser_state(
                        &mut processing_queue, &mut pending_state, new_offset, bytes.len(),
                        constraint.tokenizer.initial_state(), gss_at_offset.clone(),
                    );
                    queue_state_ns += t_qs.elapsed().as_nanos() as u64;
                    continue;
                }

                let t_adv = Instant::now();
                // Inline profiled advance (skip cache for profiling)
                let t_adv_core = Instant::now();
                let (advanced, sub_profile) = advance_stacks_profiled(
                    &constraint.table, &gss_at_offset, matched.id,
                );
                advance_core_ns += t_adv_core.elapsed().as_nanos() as u64;
                // Aggregate profile
                let t_pagg = Instant::now();
                adv_profile.n_reduces_above_floor += sub_profile.n_reduces_above_floor;
                adv_profile.n_floor_crossings += sub_profile.n_floor_crossings;
                adv_profile.n_nondet_waves += sub_profile.n_nondet_waves;
                adv_profile.n_nondet_branches += sub_profile.n_nondet_branches;
                adv_profile.clone_ns += sub_profile.clone_ns;
                adv_profile.summary_ns += sub_profile.summary_ns;
                adv_profile.fast_path_ns += sub_profile.fast_path_ns;
                adv_profile.det_ns += sub_profile.det_ns;
                adv_profile.nondet_ns += sub_profile.nondet_ns;
                if sub_profile.pure_shift { adv_profile.pure_shift = true; }
                if sub_profile.deterministic_entered { adv_profile.deterministic_entered = true; }
                if sub_profile.deterministic_finished { adv_profile.deterministic_finished = true; }
                if sub_profile.nondeterministic_entered { adv_profile.nondeterministic_entered = true; }
                adv_profile.top_states = sub_profile.top_states;
                adv_profile.vstack_len = sub_profile.vstack_len;
                adv_profile.gss_depth = sub_profile.gss_depth;
                adv_profile.det_exit_reason = sub_profile.det_exit_reason;
                adv_profile.det_exit_state = sub_profile.det_exit_state;
                adv_profile.n_det_action_lookups += sub_profile.n_det_action_lookups;
                adv_profile.n_det_goto_lookups += sub_profile.n_det_goto_lookups;
                adv_profile.n_det_popn_ops += sub_profile.n_det_popn_ops;
                adv_profile.det_action_lookup_ns += sub_profile.det_action_lookup_ns;
                adv_profile.det_goto_lookup_ns += sub_profile.det_goto_lookup_ns;
                adv_profile.det_pop_ns += sub_profile.det_pop_ns;
                adv_profile.det_push_ns += sub_profile.det_push_ns;
                adv_profile.det_floor_cross_ns += sub_profile.det_floor_cross_ns;
                adv_profile.det_floor_sources_ns += sub_profile.det_floor_sources_ns;
                adv_profile.det_floor_rebuild_ns += sub_profile.det_floor_rebuild_ns;
                adv_profile.det_floor_try_vstack_ns += sub_profile.det_floor_try_vstack_ns;
                adv_profile.n_nondet_reduce_ops += sub_profile.n_nondet_reduce_ops;
                adv_profile.n_nondet_merges += sub_profile.n_nondet_merges;
                adv_profile.n_nondet_isolates += sub_profile.n_nondet_isolates;
                adv_profile.nondet_det_ns += sub_profile.nondet_det_ns;
                profile_agg_ns += t_pagg.elapsed().as_nanos() as u64;

                let t_adv_disallow = Instant::now();
                let advanced = apply_future_terminal_disallow(
                    constraint, &exec_result, matched.id, advanced,
                );
                advance_future_disallow_ns += t_adv_disallow.elapsed().as_nanos() as u64;
                advance_ns += t_adv.elapsed().as_nanos() as u64;
                n_advances += 1;

                if advanced.is_empty() {
                    continue;
                }
                let gss = advanced;

                let t_qs = Instant::now();
                queue_parser_state(
                    &mut processing_queue, &mut pending_state, new_offset, bytes.len(),
                    constraint.tokenizer.initial_state(), gss,
                );
                queue_state_ns += t_qs.elapsed().as_nanos() as u64;
            }

            if let Some(end_state) = exec_result.end_state {
                let t_may = Instant::now();
                let may_advance = end_state_may_advance(constraint, &gss_at_offset, end_state);
                may_advance_ns += t_may.elapsed().as_nanos() as u64;
                if !may_advance {
                    continue;
                }
                let t_qs = Instant::now();
                queue_parser_state(
                    &mut processing_queue, &mut pending_state, bytes.len(), bytes.len(),
                    end_state, gss_at_offset,
                );
                queue_state_ns += t_qs.elapsed().as_nanos() as u64;
            }
        }
    }
    let queue_ns = t_queue.elapsed().as_nanos() as u64;

    let t_fuse = Instant::now();
    let (new_state, fuse_convert_ns, fuse_core_ns, fuse_retain_ns) = match pending_state.len() {
        0 => (BTreeMap::new(), 0, 0, 0),
        1 => {
            let t_fuse_convert = Instant::now();
            let (tokenizer_state, parser_state) = pending_state.into_iter().next().unwrap();
            let fuse_convert_ns = t_fuse_convert.elapsed().as_nanos() as u64;
            let t_fuse_core = Instant::now();
            let fused = parser_state.fuse(Some(1));
            let fuse_core_ns = t_fuse_core.elapsed().as_nanos() as u64;
            let t_fuse_retain = Instant::now();
            let mut retained_state = BTreeMap::new();
            if !fused.is_empty() {
                retained_state.insert(tokenizer_state, fused);
            }
            let fuse_retain_ns = t_fuse_retain.elapsed().as_nanos() as u64;
            (retained_state, fuse_convert_ns, fuse_core_ns, fuse_retain_ns)
        }
        _ => {
            let t_fuse_convert = Instant::now();
            let mut new_state: BTreeMap<u32, ParserGSS> = pending_state.into_iter().collect();
            let fuse_convert_ns = t_fuse_convert.elapsed().as_nanos() as u64;
            let t_fuse_core = Instant::now();
            for parser_state in new_state.values_mut() {
                *parser_state = parser_state.fuse(Some(1));
            }
            let fuse_core_ns = t_fuse_core.elapsed().as_nanos() as u64;
            let t_fuse_retain = Instant::now();
            new_state.retain(|_, parser_state| !parser_state.is_empty());
            let fuse_retain_ns = t_fuse_retain.elapsed().as_nanos() as u64;
            (new_state, fuse_convert_ns, fuse_core_ns, fuse_retain_ns)
        }
    };
    let fuse_ns = t_fuse.elapsed().as_nanos() as u64;

    *state = new_state;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    let total_ns = t_total.elapsed().as_nanos() as u64;
    if verbose {
        let queue_overhead = queue_ns.saturating_sub(exec_ns + advance_ns + actionable_ns + may_advance_ns + queue_state_ns + seen_matches_ns);
        eprintln!("[PROFILE_VERBOSE] === Queue Overhead Breakdown ===");
        eprintln!("[PROFILE_VERBOSE] queue_alloc_ns:       {}", queue_alloc_ns);
        eprintln!("[PROFILE_VERBOSE] queue_state_ns:       {}", queue_state_ns);
        eprintln!("[PROFILE_VERBOSE] seen_matches_ns:      {}", seen_matches_ns);
        eprintln!("[PROFILE_VERBOSE] profile_agg_ns:       {}", profile_agg_ns);
        eprintln!("[PROFILE_VERBOSE] queue_residual_ns:    {}", queue_overhead);
        eprintln!("[PROFILE_VERBOSE] n_matches_checked:    {}", n_matches_checked);
        eprintln!("[PROFILE_VERBOSE] n_matches_skip_act:   {}", n_matches_skipped_actionable);
        eprintln!("[PROFILE_VERBOSE] n_matches_skip_seen:  {}", n_matches_skipped_seen);
        eprintln!("[PROFILE_VERBOSE] n_ignored:            {}", n_ignored);
        eprintln!("[PROFILE_VERBOSE] === Fuse Breakdown ===");
        eprintln!("[PROFILE_VERBOSE] fuse_convert_ns:      {}", fuse_convert_ns);
        eprintln!("[PROFILE_VERBOSE] fuse_core_ns:         {}", fuse_core_ns);
        eprintln!("[PROFILE_VERBOSE] fuse_retain_ns:       {}", fuse_retain_ns);
        eprintln!("[PROFILE_VERBOSE] fuse_total_ns:        {}", fuse_ns);
        eprintln!("[PROFILE_VERBOSE] n_pending_entries:    {}", state.len());
    }

    Ok((total_ns, scan_ns, prune_ns, queue_ns, fuse_ns, exec_ns, advance_ns,
        advance_may_check_ns, advance_core_ns, advance_future_disallow_ns,
        actionable_ns, may_advance_ns,
        n_tokenizer_states, n_queue_entries, n_advances,
        adv_profile.n_reduces_above_floor as u64,
        adv_profile.n_floor_crossings as u64,
        adv_profile.n_nondet_waves as u64,
        adv_profile.n_nondet_branches as u64,
        adv_profile.clone_ns,
        adv_profile.fast_path_ns,
        adv_profile.det_ns,
        adv_profile.nondet_ns,
        adv_profile.vstack_len as u64,
        adv_profile.gss_depth as u64,
        adv_profile.det_exit_reason as u64,
        adv_profile.det_exit_state as u64,
        adv_profile.summary_ns,
        adv_profile.n_det_action_lookups as u64,
        adv_profile.n_det_goto_lookups as u64,
        adv_profile.n_det_popn_ops as u64,
        adv_profile.det_action_lookup_ns,
        adv_profile.det_goto_lookup_ns,
        adv_profile.det_pop_ns,
        adv_profile.det_push_ns,
        adv_profile.det_floor_cross_ns,
        adv_profile.det_floor_sources_ns,
        adv_profile.det_floor_rebuild_ns,
        adv_profile.det_floor_try_vstack_ns,
        adv_profile.n_nondet_reduce_ops as u64,
        adv_profile.n_nondet_merges as u64,
        adv_profile.n_nondet_isolates as u64,
        adv_profile.nondet_det_ns))
}

/// Per-advance entry capturing GSS state and advance profile.
pub struct PerAdvanceEntry {
    pub terminal_id: u32,
    pub tokenizer_state: u32,
    pub gss_stacks_before: Vec<Vec<u32>>,
    pub gss_stacks_after: Vec<Vec<u32>>,
    pub gss_summary_upperbranch_before: usize,
    pub gss_summary_interface_before: usize,
    pub gss_summary_lower_before: usize,
    pub gss_summary_lower_general_before: usize,
    pub gss_summary_lower_segment_before: usize,
    pub gss_summary_edges_before: usize,
    pub gss_summary_depth_before: u32,
    pub gss_summary_upperbranch_after: usize,
    pub gss_summary_interface_after: usize,
    pub gss_summary_lower_after: usize,
    pub gss_summary_lower_general_after: usize,
    pub gss_summary_lower_segment_after: usize,
    pub gss_summary_edges_after: usize,
    pub gss_summary_depth_after: u32,
    /// Start byte offset (inclusive) within the committed token bytes.
    pub match_start: usize,
    /// End byte offset (exclusive) within the committed token bytes.
    pub match_end: usize,
    /// Total committed token byte length (position bound).
    pub token_bound: usize,
    /// Actual token-byte slice consumed by this tokenizer match.
    pub match_bytes: Vec<u8>,
    pub profile: AdvanceProfile,
}

/// Like commit_bytes_impl_profiled but returns per-advance entries instead of aggregated timing.
fn commit_bytes_per_advance(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>), String> {
    if bytes.is_empty() {
        let final_stacks = state.iter().map(|(&ts, gss)| {
            (ts, gss.to_stacks().into_iter().map(|(s, _)| s).collect())
        }).collect();
        return Ok((Vec::new(), final_stacks));
    }

    let ignore_terminal = constraint.ignore_terminal;

    let mut initial_scan = InitialCommitScan::collect(constraint, state, bytes);
    prune_initial_states(
        state,
        &initial_scan.accepted_terminals,
        &initial_scan.remapped_tokenizer_states,
    );
    state.retain(|_, parser_state| !parser_state.is_empty());

    let mut pending_state = ParserStatesByTokenizer::default();
    let mut processing_queue: Vec<ParserStatesByTokenizer> =
        (0..=bytes.len()).map(|_| ParserStatesByTokenizer::default()).collect();
    processing_queue[0] = std::mem::take(state).into_iter().collect();

    let mut advances = Vec::new();
    let mut offset = 0usize;

    while offset < processing_queue.len() {
        if processing_queue[offset].is_empty() {
            offset += 1;
            continue;
        }

        let states_to_process = std::mem::take(&mut processing_queue[offset]);
        for (tokenizer_state, gss_at_offset) in states_to_process {
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            let exec_result = if offset == 0 {
                initial_scan.take_exec_result(tokenizer_state).unwrap_or_else(|| {
                    constraint.tokenizer.execute_from_state(&bytes[offset..], tokenizer_state)
                })
            } else {
                constraint.tokenizer.execute_from_state(&bytes[offset..], tokenizer_state)
            };

            let mut seen_matches = FxHashSet::default();

            for matched in &exec_result.matches {
                let new_offset = offset + matched.width;
                let ignored = is_ignored_terminal(ignore_terminal, matched.id);

                if !ignored
                    && !is_actionable_terminal(actionable_terminals.as_ref(), constraint, matched.id)
                {
                    continue;
                }
                if !seen_matches.insert((matched.width, matched.id)) {
                    continue;
                }

                if ignored {
                    queue_parser_state(
                        &mut processing_queue, &mut pending_state, new_offset, bytes.len(),
                        constraint.tokenizer.initial_state(), gss_at_offset.clone(),
                    );
                    continue;
                }

                if !stack_may_advance_on(&constraint.table, &gss_at_offset, matched.id) {
                    continue;
                }

                // Capture GSS stacks and summary before advance.
                let gss_stacks_before: Vec<Vec<u32>> = gss_at_offset.to_stacks()
                    .into_iter().map(|(s, _)| s).collect();
                let gss_summary_before = gss_at_offset.summary();

                let (advanced, sub_profile) = advance_stacks_profiled(
                    &constraint.table, &gss_at_offset, matched.id,
                );

                let advanced = apply_future_terminal_disallow(
                    constraint, &exec_result, matched.id, advanced,
                );

                // Capture GSS stacks and summary after advance/disallow.
                let gss_stacks_after: Vec<Vec<u32>> = advanced.to_stacks()
                    .into_iter().map(|(s, _)| s).collect();
                let gss_summary_after = advanced.summary();

                advances.push(PerAdvanceEntry {
                    terminal_id: matched.id,
                    tokenizer_state,
                    gss_stacks_before,
                    gss_stacks_after,
                    gss_summary_upperbranch_before: gss_summary_before.upperbranch_nodes,
                    gss_summary_interface_before: gss_summary_before.interface_nodes,
                    gss_summary_lower_before: gss_summary_before.lower_nodes,
                    gss_summary_lower_general_before: gss_summary_before.lower_general_nodes,
                    gss_summary_lower_segment_before: gss_summary_before.lower_segment_nodes,
                    gss_summary_edges_before: gss_summary_before.total_edges,
                    gss_summary_depth_before: gss_summary_before.max_depth,
                    gss_summary_upperbranch_after: gss_summary_after.upperbranch_nodes,
                    gss_summary_interface_after: gss_summary_after.interface_nodes,
                    gss_summary_lower_after: gss_summary_after.lower_nodes,
                    gss_summary_lower_general_after: gss_summary_after.lower_general_nodes,
                    gss_summary_lower_segment_after: gss_summary_after.lower_segment_nodes,
                    gss_summary_edges_after: gss_summary_after.total_edges,
                    gss_summary_depth_after: gss_summary_after.max_depth,
                    match_start: offset,
                    match_end: new_offset,
                    token_bound: bytes.len(),
                    match_bytes: bytes[offset..new_offset].to_vec(),
                    profile: sub_profile,
                });

                if advanced.is_empty() {
                    continue;
                }

                queue_parser_state(
                    &mut processing_queue, &mut pending_state, new_offset, bytes.len(),
                    constraint.tokenizer.initial_state(), advanced,
                );
            }

            if let Some(end_state) = exec_result.end_state {
                if end_state_may_advance(constraint, &gss_at_offset, end_state) {
                    queue_parser_state(
                        &mut processing_queue, &mut pending_state, bytes.len(), bytes.len(),
                        end_state, gss_at_offset.clone(),
                    );
                }
            }
        }
        offset += 1;
    }

    // Fuse
    let new_state = finalize_pending_state(pending_state);

    let final_stacks: Vec<(u32, Vec<Vec<u32>>)> = new_state.iter().map(|(&ts, gss)| {
        (ts, gss.to_stacks().into_iter().map(|(s, _)| s).collect())
    }).collect();

    *state = new_state;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    Ok((advances, final_stacks))
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

    pub fn commit_token_timed_ns(
        &mut self,
        token_id: u32,
    ) -> Result<u64, String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let t_start = std::time::Instant::now();
        let result = commit_bytes_impl(constraint, &mut self.state, bytes, &mut self.buffers);
        let elapsed_ns = t_start.elapsed().as_nanos() as u64;
        self.generation += 1;
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result.map(|()| elapsed_ns)
    }

    /// Like commit_token but returns profiling stats.
    /// Returns 22-tuple with timing and count metrics.
    pub fn commit_token_profiled(
        &mut self,
        token_id: u32,
    ) -> Result<(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let result = commit_bytes_impl_profiled(constraint, &mut self.state, bytes);
        assert_mask_commit_equivalence(token_id, bytes, was_in_mask, result.is_ok());
        result
    }

    /// Like commit_token_profiled but returns per-advance entries.
    /// Returns (advances, final_gss_stacks) where each advance entry contains
    /// terminal_id, tokenizer_state, GSS stacks before advance, and advance profile.
    pub fn commit_token_per_advance(
        &mut self,
        token_id: u32,
    ) -> Result<(Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        let was_in_mask = snapshot_mask_membership(self, token_id);
        let result = commit_bytes_per_advance(constraint, &mut self.state, bytes);
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
