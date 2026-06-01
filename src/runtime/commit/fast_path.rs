//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

pub(super) fn commit_bytes_fast_path(
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
        if !stack_can_advance_on(&constraint.table, gss, matched.id) {
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
        .filter(|&end_state| end_state_can_advance(constraint, &pruned_gss, end_state));
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

pub(super) fn commit_bytes_full_width_fast_path(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Option<Result<(), String>> {
    if state.len() > 2 {
        return None;
    }
    if state.len() > 1 && state_has_nonempty_accumulators(state) {
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
                if !stack_can_advance_on(&constraint.table, &pruned_gss, terminal) {
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
            if end_state_can_advance(constraint, &pruned_gss, end_state) {
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

pub(super) fn commit_bytes_small_queue_fast_path(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
) -> Option<Result<(), String>> {
    if bytes.len() > 8 || state.len() > 2 {
        return None;
    }
    if state.len() > 1 && state_has_nonempty_accumulators(state) {
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

            for matched in normalized_matches {
                let new_offset = offset + matched.width;
                if new_offset > bytes.len() {
                    return None;
                }

                if matched.ignored {
                    if new_offset == bytes.len() {
                        merge_parser_state(
                            &mut pending_state,
                            constraint.tokenizer.initial_state(),
                            gss_at_offset.clone(),
                        );
                    } else {
                        merge_small_parser_state(
                            &mut processing_queue[new_offset],
                            constraint.tokenizer.initial_state(),
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
                    if !stack_can_advance_on(
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
                if new_offset == bytes.len() {
                    merge_parser_state(
                        &mut pending_state,
                        constraint.tokenizer.initial_state(),
                        advanced,
                    );
                } else {
                    merge_small_parser_state(
                        &mut processing_queue[new_offset],
                        constraint.tokenizer.initial_state(),
                        advanced,
                    );
                }
            }

            if let Some(end_state) = exec_result.end_state {
                if end_state_can_advance(constraint, &gss_at_offset, end_state) {
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

pub(super) fn choose_direct_linear_step(
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

pub(super) fn commit_bytes_direct_linear_fast_path(
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
            let should_restart = end_state_can_advance(constraint, &gss, end_state);
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

pub(super) fn commit_bytes_linear_fast_path(
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
            if end_state_can_advance(constraint, &gss, end_state) {
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

