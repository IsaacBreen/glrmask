//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

pub(super) fn parser_stacks_only(gss: &ParserGSS) -> Vec<Vec<u32>> {
    gss.to_stacks().into_iter().map(|(stack, _)| stack).collect()
}

pub(super) fn record_per_advance_entry(
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

pub(super) fn commit_bytes_fast_path_profiled(
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
        if !stack_can_advance_on(&constraint.table, gss, matched.id) {
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
        .filter(|&end_state| end_state_can_advance(constraint, &pruned_gss, end_state));
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

pub(super) fn commit_bytes_impl_profiled(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
    bufs: &mut CommitBuffers,
    mut advances: Option<&mut Vec<PerAdvanceEntry>>,
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

    if state.len() == 1 {
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
                    .is_some_and(|gss| end_state_can_advance(constraint, gss, end_state))
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
                                let can_advance =
                                    stack_can_advance_on(&constraint.table, &gss_at_offset, matched.terminal_id);
                                let may_elapsed = may_start.elapsed().as_nanos() as u64;
                                profile.advance_may_check_ns += may_elapsed;
                                if !can_advance {
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
                                let can_advance =
                                    end_state_can_advance(constraint, &gss_at_offset, end_state);
                                profile.may_advance_ns += may_start.elapsed().as_nanos() as u64;
                                if !can_advance {
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
                let can_advance =
                    stack_can_advance_on(&constraint.table, &gss_at_offset, matched.terminal_id);
                let may_elapsed = may_start.elapsed().as_nanos() as u64;
                profile.advance_may_check_ns += may_elapsed;
                if !can_advance {
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
                let can_advance = end_state_can_advance(constraint, &gss_at_offset, end_state);
                profile.may_advance_ns += may_start.elapsed().as_nanos() as u64;
                if !can_advance {
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

pub(super) fn final_stacks(state: &BTreeMap<u32, ParserGSS>) -> Vec<(u32, Vec<Vec<u32>>)> {
    state.iter().map(|(&tokenizer_state, gss)| {
        (tokenizer_state, parser_stacks_only(gss))
    }).collect()
}

pub(super) fn commit_bytes_linear_fast_path_profiled(
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
            if end_state_can_advance(constraint, &gss, end_state) {
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

