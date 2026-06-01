//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

pub(super) fn commit_bytes_impl(
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
                    .is_some_and(|gss| end_state_can_advance(constraint, gss, end_state))
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
                                if !end_state_can_advance(constraint, &gss_at_offset, end_state) {
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
                if !end_state_can_advance(constraint, &gss_at_offset, end_state) {
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

