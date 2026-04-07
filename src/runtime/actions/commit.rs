use std::collections::{BTreeMap, BTreeSet};

use crate::automata::lexer::tokenizer::TokenizerExecResult;
use crate::compiler::glr::parser::{
    AdvanceProfile,
    ParserGSS,
    TerminalsDisallowed,
    advance_stacks,
    advance_stacks_profiled,
    stack_may_advance_on,
    stack_may_advance_on_any,
};
use crate::compiler::glr::table::Action;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::ConstraintState;
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
        if !stack_may_advance_on(&constraint.table, gss_at_offset, terminal) {
            let empty = ParserGSS::empty();
            advance_result_cache.insert(advance_cache_key, (gss_at_offset.clone(), empty.clone()));
            terminal_result_cache.insert(terminal, empty);
            return None;
        }

        let advanced = advance_stacks(&constraint.table, gss_at_offset, terminal);
        advance_result_cache.insert(advance_cache_key, (gss_at_offset.clone(), advanced.clone()));
        advanced
    };

    let advanced = apply_future_terminal_disallow(constraint, exec_result, terminal, advanced);
    terminal_result_cache.insert(terminal, advanced.clone());
    (!advanced.is_empty()).then_some(advanced)
}

fn advance_terminal_match_profiled(
    constraint: &Constraint,
    gss_at_offset: &ParserGSS,
    terminal: u32,
    exec_result: &TokenizerExecResult,
    advance_result_cache: &mut AdvanceResultCache,
    terminal_result_cache: &mut FxHashMap<u32, ParserGSS>,
    adv_profile: &mut AdvanceProfile,
) -> Option<ParserGSS> {
    if let Some(cached) = terminal_result_cache.get(&terminal) {
        return (!cached.is_empty()).then(|| cached.clone());
    }

    let advance_cache_key = (gss_at_offset.ptr_key(), terminal);
    let advanced = if let Some((_, cached)) = advance_result_cache.get(&advance_cache_key) {
        cached.clone()
    } else {
        if !stack_may_advance_on(&constraint.table, gss_at_offset, terminal) {
            let empty = ParserGSS::empty();
            advance_result_cache.insert(advance_cache_key, (gss_at_offset.clone(), empty.clone()));
            terminal_result_cache.insert(terminal, empty);
            return None;
        }

        let (advanced, profile) = advance_stacks_profiled(&constraint.table, gss_at_offset, terminal);
        adv_profile.isolate_ns += profile.isolate_ns;
        adv_profile.popn_ns += profile.popn_ns;
        adv_profile.base_isolate_ns += profile.base_isolate_ns;
        adv_profile.merge_ns += profile.merge_ns;
        adv_profile.absorb_push_ns += profile.absorb_push_ns;
        adv_profile.shift_ns += profile.shift_ns;
        adv_profile.n_loop_iters += profile.n_loop_iters;
        adv_profile.n_reduces += profile.n_reduces;
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
fn commit_bytes_fast_path(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Option<Result<(), String>> {
    if bytes.is_empty() || state.len() != 1 {
        return None;
    }

    let (&tokenizer_state, gss) = state.iter().next().unwrap();
    let ignore_terminal = constraint.ignore_terminal;

    // Execute tokenizer once
    let exec_result = constraint.tokenizer.execute_from_state(bytes, tokenizer_state);

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
        let future_terminals = constraint.tokenizer.possible_future_terminals(end_state);
        if stack_may_advance_on_any(&constraint.table, gss, future_terminals) {
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
            if let Some(Action::Shift(target)) = constraint.table.action(top_state, terminal) {
                // Single state, single terminal, no end_state, empty accs, pure shift.
                // Prune is identity (empty accs → nothing disallowed, no remap).
                // advance_stacks → push(target).
                // fuse(1) on push result → identity (Interface, no multi-depth slots).
                // apply_future_terminal_disallow → identity (no end_state).
                let shifted = gss.push(*target);
                state.clear();
                state.insert(constraint.tokenizer.initial_state(), shifted);
                return Some(Ok(()));
            }
        }
    }

    // Standard fast path: skip prune when accumulators are empty.
    let pruned_gss = if all_accs_empty {
        gss.clone()
    } else {
        let pruned = gss.apply_and_prune_no_promote(|td: &TerminalsDisallowed| {
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

    // Advance the parser
    let advanced = advance_stacks(&constraint.table, &pruned_gss, terminal);
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

    state.clear();
    state.insert(constraint.tokenizer.initial_state(), fused);
    Some(Ok(()))
}

fn commit_bytes_impl(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }

    // Try the fast path first (single tokenizer state, single terminal match)
    if let Some(result) = commit_bytes_fast_path(constraint, state, bytes) {
        return result;
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
    let mut advance_result_cache = AdvanceResultCache::default();
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
            let actionable_terminals = ActionableTerminals::from_gss(constraint, &gss_at_offset);
            let exec_result = if offset == 0 {
                initial_scan.take_exec_result(tokenizer_state).unwrap_or_else(|| {
                    constraint
                        .tokenizer
                        .execute_from_state(&bytes[offset..], tokenizer_state)
                })
            } else {
                constraint
                    .tokenizer
                    .execute_from_state(&bytes[offset..], tokenizer_state)
            };

            let mut seen_matches = FxHashSet::default();
            let mut terminal_result_cache = FxHashMap::<u32, ParserGSS>::default();

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
                if !seen_matches.insert((matched.width, matched.id)) {
                    continue;
                }

                if ignored {
                    queue_parser_state(
                        &mut processing_queue,
                        &mut pending_state,
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
                    &mut advance_result_cache,
                    &mut terminal_result_cache,
                ) else {
                    continue;
                };

                queue_parser_state(
                    &mut processing_queue,
                    &mut pending_state,
                    new_offset,
                    bytes.len(),
                    constraint.tokenizer.initial_state(),
                    gss,
                );
            }

            if let Some(end_state) = exec_result.end_state {
                let future_terminals = constraint.tokenizer.possible_future_terminals(end_state);
                if !stack_may_advance_on_any(&constraint.table, &gss_at_offset, future_terminals)
                {
                    continue;
                }

                queue_parser_state(
                    &mut processing_queue,
                    &mut pending_state,
                    bytes.len(),
                    bytes.len(),
                    end_state,
                    gss_at_offset,
                );
            }
        }
    }

    let mut new_state: BTreeMap<u32, ParserGSS> = pending_state.into_iter().collect();
    for parser_state in new_state.values_mut() {
        *parser_state = parser_state.fuse(Some(1));
    }
    new_state.retain(|_, parser_state| !parser_state.is_empty());

    *state = new_state;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    Ok(())
}

fn commit_bytes_impl_profiled(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Result<(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64), String> {
    use std::time::Instant;
    let t_total = Instant::now();

    if bytes.is_empty() {
        let total_ns = t_total.elapsed().as_nanos() as u64;
        return Ok((total_ns, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
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
    let mut pending_state = ParserStatesByTokenizer::default();
    let mut advance_result_cache = AdvanceResultCache::default();
    let mut processing_queue: Vec<ParserStatesByTokenizer> =
        (0..=bytes.len()).map(|_| ParserStatesByTokenizer::default()).collect();
    processing_queue[0] = std::mem::take(state).into_iter().collect();

    let mut n_queue_entries: u64 = 0;
    let mut exec_ns: u64 = 0;
    let mut advance_ns: u64 = 0;
    let mut actionable_ns: u64 = 0;
    let mut may_advance_ns: u64 = 0;
    let mut n_advances: u64 = 0;
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
            let mut terminal_result_cache = FxHashMap::<u32, ParserGSS>::default();

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

                let t_adv = Instant::now();
                let advance_result = advance_terminal_match_profiled(
                    constraint, &gss_at_offset, matched.id, &exec_result,
                    &mut advance_result_cache, &mut terminal_result_cache,
                    &mut adv_profile,
                );
                advance_ns += t_adv.elapsed().as_nanos() as u64;
                n_advances += 1;

                let Some(gss) = advance_result else {
                    continue;
                };

                queue_parser_state(
                    &mut processing_queue, &mut pending_state, new_offset, bytes.len(),
                    constraint.tokenizer.initial_state(), gss,
                );
            }

            if let Some(end_state) = exec_result.end_state {
                let future_terminals = constraint.tokenizer.possible_future_terminals(end_state);
                let t_may = Instant::now();
                let may_advance = stack_may_advance_on_any(&constraint.table, &gss_at_offset, future_terminals);
                may_advance_ns += t_may.elapsed().as_nanos() as u64;
                if !may_advance {
                    continue;
                }
                queue_parser_state(
                    &mut processing_queue, &mut pending_state, bytes.len(), bytes.len(),
                    end_state, gss_at_offset,
                );
            }
        }
    }
    let queue_ns = t_queue.elapsed().as_nanos() as u64;

    let t_fuse = Instant::now();
    let mut new_state: BTreeMap<u32, ParserGSS> = pending_state.into_iter().collect();
    for parser_state in new_state.values_mut() {
        *parser_state = parser_state.fuse(Some(1));
    }
    new_state.retain(|_, parser_state| !parser_state.is_empty());
    let fuse_ns = t_fuse.elapsed().as_nanos() as u64;

    *state = new_state;
    if state.is_empty() {
        return Err("commit rejected: no valid parser states remain".to_string());
    }

    let total_ns = t_total.elapsed().as_nanos() as u64;
    Ok((total_ns, scan_ns, prune_ns, queue_ns, fuse_ns, exec_ns, advance_ns,
        actionable_ns, may_advance_ns,
        n_tokenizer_states, n_queue_entries, n_advances,
        adv_profile.isolate_ns, adv_profile.popn_ns, adv_profile.base_isolate_ns,
        adv_profile.merge_ns, adv_profile.absorb_push_ns, adv_profile.shift_ns,
        adv_profile.n_loop_iters as u64, adv_profile.n_reduces as u64))
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
        commit_bytes_impl(constraint, &mut self.state, bytes)
    }

    /// Like commit_token but returns profiling stats.
    /// Returns 20-tuple with timing and count metrics.
    pub fn commit_token_profiled(
        &mut self,
        token_id: u32,
    ) -> Result<(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64), String> {
        let constraint = self.constraint;
        let bytes = token_bytes_for_id(constraint, token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        commit_bytes_impl_profiled(constraint, &mut self.state, bytes)
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        commit_bytes_impl(self.constraint, &mut self.state, bytes)
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token(token)?;
        }
        Ok(())
    }
}
