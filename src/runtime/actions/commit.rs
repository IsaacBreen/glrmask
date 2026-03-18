use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::glr::parser::{
    AdvanceStacksDebugMetrics,
    ParserGSS,
    TerminalsDisallowed,
    advance_stacks_with_metrics,
    stack_may_advance_on,
};
use crate::ds::leveled_gss::LeveledGSSSummary;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::{ConstraintState, ConstraintStateSummary};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommitDebugMetrics {
    pub bytes_len: usize,
    pub state_summary_before: ConstraintStateSummary,
    pub state_summary_after: ConstraintStateSummary,
    pub initial_tokenizer_states: usize,
    pub initial_exec_calls: usize,
    pub initial_exec_end_state_hits: usize,
    pub initial_matches_total: usize,
    pub initial_ignored_matches: usize,
    pub initial_terminals_total: usize,
    pub initial_terminals_map_entries: usize,
    pub remapped_state_entries: usize,
    pub parser_states_pruned: usize,
    pub parser_states_retained_after_prune: usize,
    pub queue_offsets_processed: usize,
    pub queue_states_processed: usize,
    pub queue_max_offsets_pending: usize,
    pub queue_max_states_in_offset_bucket: usize,
    pub processing_exec_calls: usize,
    pub reused_initial_exec_results: usize,
    pub processing_matches_total: usize,
    pub processing_ignored_matches: usize,
    pub advance_stacks_calls: usize,
    pub advance_stacks_nonempty: usize,
    pub advance_reduce_closure_iterations_total: usize,
    pub advance_reduce_closure_iterations_max: usize,
    pub advance_frontier_states_total: usize,
    pub advance_frontier_states_max: usize,
    pub advance_reduce_rules_considered: usize,
    pub advance_popn_calls: usize,
    pub advance_popn_nonempty: usize,
    pub advance_goto_lookups: usize,
    pub advance_goto_hits: usize,
    pub advance_reductions_emitted: usize,
    pub advance_absorb_targets: usize,
    pub advance_shift_state_candidates: usize,
    pub advance_shift_targets_hit: usize,
    pub advance_shifted_results: usize,
    pub advance_reduce_rule_considered_counts: BTreeMap<u32, usize>,
    pub advance_reduce_rule_emitted_counts: BTreeMap<u32, usize>,
    pub advance_reduce_rhs_len_emitted_counts: BTreeMap<usize, usize>,
    pub advance_reduce_lhs_emitted_counts: BTreeMap<u32, usize>,
    pub advance_reduce_state_emitted_counts: BTreeMap<u32, usize>,
    pub advance_goto_from_counts: BTreeMap<u32, usize>,
    pub advance_goto_target_counts: BTreeMap<u32, usize>,
    pub advance_input_top_values_total: usize,
    pub advance_input_top_values_max: usize,
    pub advance_input_upperbranch_nodes_total: usize,
    pub advance_input_upperbranch_nodes_max: usize,
    pub advance_input_interface_nodes_total: usize,
    pub advance_input_interface_nodes_max: usize,
    pub advance_input_lower_nodes_total: usize,
    pub advance_input_lower_nodes_max: usize,
    pub advance_input_unique_nodes_total: usize,
    pub advance_input_unique_nodes_max: usize,
    pub advance_input_total_edges_total: usize,
    pub advance_input_total_edges_max: usize,
    pub advance_output_top_values_total: usize,
    pub advance_output_top_values_max: usize,
    pub advance_output_upperbranch_nodes_total: usize,
    pub advance_output_upperbranch_nodes_max: usize,
    pub advance_output_interface_nodes_total: usize,
    pub advance_output_interface_nodes_max: usize,
    pub advance_output_lower_nodes_total: usize,
    pub advance_output_lower_nodes_max: usize,
    pub advance_output_unique_nodes_total: usize,
    pub advance_output_unique_nodes_max: usize,
    pub advance_output_total_edges_total: usize,
    pub advance_output_total_edges_max: usize,
    pub future_group_checks: usize,
    pub future_group_hits: usize,
    pub future_group_updates: usize,
    pub ignored_terminal_queue_pushes: usize,
    pub ignored_terminal_queue_merges: usize,
    pub ignored_terminal_final_pushes: usize,
    pub ignored_terminal_final_merges: usize,
    pub parser_queue_pushes: usize,
    pub parser_queue_merges: usize,
    pub parser_final_pushes: usize,
    pub parser_final_merges: usize,
    pub passthrough_end_state_pushes: usize,
    pub passthrough_end_state_merges: usize,
    pub fused_parser_states: usize,
    pub initial_tokenizer_exec_ns: u64,
    pub initial_apply_prune_ns: u64,
    pub initial_remap_ns: u64,
    pub processing_tokenizer_exec_ns: u64,
    pub advance_stacks_ns: u64,
    pub future_group_apply_ns: u64,
    pub merge_ns: u64,
    pub fuse_ns: u64,
    pub total_ns: u64,
}

fn summarize_state_map(state: &BTreeMap<u32, ParserGSS>) -> ConstraintStateSummary {
    let mut summary = ConstraintStateSummary {
        tokenizer_state_count: state.len(),
        ..ConstraintStateSummary::default()
    };

    for gss in state.values() {
        if gss.is_empty() {
            continue;
        }

        summary.nonempty_tokenizer_state_count += 1;
        let gss_summary: LeveledGSSSummary = gss.summary();
        summary.parser_top_values_total += gss_summary.top_values_count;
        summary.parser_top_values_max = summary
            .parser_top_values_max
            .max(gss_summary.top_values_count);
        summary.parser_upperbranch_nodes_total += gss_summary.upperbranch_nodes;
        summary.parser_upperbranch_nodes_max = summary
            .parser_upperbranch_nodes_max
            .max(gss_summary.upperbranch_nodes);
        summary.parser_interface_nodes_total += gss_summary.interface_nodes;
        summary.parser_interface_nodes_max = summary
            .parser_interface_nodes_max
            .max(gss_summary.interface_nodes);
        summary.parser_lower_nodes_total += gss_summary.lower_nodes;
        summary.parser_lower_nodes_max = summary
            .parser_lower_nodes_max
            .max(gss_summary.lower_nodes);
        summary.parser_unique_nodes_total += gss_summary.total_unique_nodes;
        summary.parser_unique_nodes_max = summary
            .parser_unique_nodes_max
            .max(gss_summary.total_unique_nodes);
        summary.parser_total_edges_total += gss_summary.total_edges;
        summary.parser_accumulator_instances_total += gss_summary.accumulator_instances;
        summary.parser_max_depth = summary.parser_max_depth.max(gss_summary.max_depth);
    }

    summary
}

fn accumulate_advance_stacks_metrics(
    metrics: &mut CommitDebugMetrics,
    advance_metrics: &AdvanceStacksDebugMetrics,
) {
    fn merge_counts<K: Ord + Copy>(
        dst: &mut BTreeMap<K, usize>,
        src: &BTreeMap<K, usize>,
    ) {
        for (&key, &count) in src {
            *dst.entry(key).or_default() += count;
        }
    }

    metrics.advance_reduce_closure_iterations_total += advance_metrics.reduce_closure_iterations;
    metrics.advance_reduce_closure_iterations_max = metrics
        .advance_reduce_closure_iterations_max
        .max(advance_metrics.reduce_closure_iterations);
    metrics.advance_frontier_states_total += advance_metrics.frontier_states_total;
    metrics.advance_frontier_states_max = metrics
        .advance_frontier_states_max
        .max(advance_metrics.frontier_states_max);
    metrics.advance_reduce_rules_considered += advance_metrics.reduce_rules_considered;
    metrics.advance_popn_calls += advance_metrics.popn_calls;
    metrics.advance_popn_nonempty += advance_metrics.popn_nonempty;
    metrics.advance_goto_lookups += advance_metrics.goto_lookups;
    metrics.advance_goto_hits += advance_metrics.goto_hits;
    metrics.advance_reductions_emitted += advance_metrics.reductions_emitted;
    metrics.advance_absorb_targets += advance_metrics.absorb_targets;
    metrics.advance_shift_state_candidates += advance_metrics.shift_state_candidates;
    metrics.advance_shift_targets_hit += advance_metrics.shift_targets_hit;
    metrics.advance_shifted_results += advance_metrics.shifted_results;
    merge_counts(
        &mut metrics.advance_reduce_rule_considered_counts,
        &advance_metrics.reduce_rule_considered_counts,
    );
    merge_counts(
        &mut metrics.advance_reduce_rule_emitted_counts,
        &advance_metrics.reduce_rule_emitted_counts,
    );
    merge_counts(
        &mut metrics.advance_reduce_rhs_len_emitted_counts,
        &advance_metrics.reduce_rhs_len_emitted_counts,
    );
    merge_counts(
        &mut metrics.advance_reduce_lhs_emitted_counts,
        &advance_metrics.reduce_lhs_emitted_counts,
    );
    merge_counts(
        &mut metrics.advance_reduce_state_emitted_counts,
        &advance_metrics.reduce_state_emitted_counts,
    );
    merge_counts(
        &mut metrics.advance_goto_from_counts,
        &advance_metrics.goto_from_counts,
    );
    merge_counts(
        &mut metrics.advance_goto_target_counts,
        &advance_metrics.goto_target_counts,
    );

    metrics.advance_input_top_values_total += advance_metrics.input_summary.top_values_count;
    metrics.advance_input_top_values_max = metrics
        .advance_input_top_values_max
        .max(advance_metrics.input_summary.top_values_count);
    metrics.advance_input_upperbranch_nodes_total += advance_metrics.input_summary.upperbranch_nodes;
    metrics.advance_input_upperbranch_nodes_max = metrics
        .advance_input_upperbranch_nodes_max
        .max(advance_metrics.input_summary.upperbranch_nodes);
    metrics.advance_input_interface_nodes_total += advance_metrics.input_summary.interface_nodes;
    metrics.advance_input_interface_nodes_max = metrics
        .advance_input_interface_nodes_max
        .max(advance_metrics.input_summary.interface_nodes);
    metrics.advance_input_lower_nodes_total += advance_metrics.input_summary.lower_nodes;
    metrics.advance_input_lower_nodes_max = metrics
        .advance_input_lower_nodes_max
        .max(advance_metrics.input_summary.lower_nodes);
    metrics.advance_input_unique_nodes_total += advance_metrics.input_summary.total_unique_nodes;
    metrics.advance_input_unique_nodes_max = metrics
        .advance_input_unique_nodes_max
        .max(advance_metrics.input_summary.total_unique_nodes);
    metrics.advance_input_total_edges_total += advance_metrics.input_summary.total_edges;
    metrics.advance_input_total_edges_max = metrics
        .advance_input_total_edges_max
        .max(advance_metrics.input_summary.total_edges);

    metrics.advance_output_top_values_total += advance_metrics.output_summary.top_values_count;
    metrics.advance_output_top_values_max = metrics
        .advance_output_top_values_max
        .max(advance_metrics.output_summary.top_values_count);
    metrics.advance_output_upperbranch_nodes_total += advance_metrics.output_summary.upperbranch_nodes;
    metrics.advance_output_upperbranch_nodes_max = metrics
        .advance_output_upperbranch_nodes_max
        .max(advance_metrics.output_summary.upperbranch_nodes);
    metrics.advance_output_interface_nodes_total += advance_metrics.output_summary.interface_nodes;
    metrics.advance_output_interface_nodes_max = metrics
        .advance_output_interface_nodes_max
        .max(advance_metrics.output_summary.interface_nodes);
    metrics.advance_output_lower_nodes_total += advance_metrics.output_summary.lower_nodes;
    metrics.advance_output_lower_nodes_max = metrics
        .advance_output_lower_nodes_max
        .max(advance_metrics.output_summary.lower_nodes);
    metrics.advance_output_unique_nodes_total += advance_metrics.output_summary.total_unique_nodes;
    metrics.advance_output_unique_nodes_max = metrics
        .advance_output_unique_nodes_max
        .max(advance_metrics.output_summary.total_unique_nodes);
    metrics.advance_output_total_edges_total += advance_metrics.output_summary.total_edges;
    metrics.advance_output_total_edges_max = metrics
        .advance_output_total_edges_max
        .max(advance_metrics.output_summary.total_edges);
}

fn commit_bytes_impl(
    constraint: &Constraint,
    state: &mut BTreeMap<u32, ParserGSS>,
    bytes: &[u8],
    mut metrics: Option<&mut CommitDebugMetrics>,
) {
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.bytes_len = bytes.len();
    }

    if bytes.is_empty() {
        return;
    }

    let t_total = metrics
        .as_ref()
        .map(|_| std::time::Instant::now());
    let ignore_terminal = constraint.ignore_terminal;
    let mut initial_exec_results = BTreeMap::new();
    let mut state_map = BTreeMap::new();
    let mut terminals_map = BTreeMap::<u32, BTreeSet<u32>>::new();

    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.initial_tokenizer_states = state.len();
    }

    for &tokenizer_state in state.keys() {
        let t_exec = metrics
            .as_ref()
            .map(|_| std::time::Instant::now());
        let exec = constraint.tokenizer.execute_from_state(bytes, tokenizer_state);
        if let (Some(metrics), Some(t_exec)) = (metrics.as_deref_mut(), t_exec) {
            metrics.initial_exec_calls += 1;
            metrics.initial_tokenizer_exec_ns += t_exec.elapsed().as_nanos() as u64;
            metrics.initial_matches_total += exec.matches.len();
            if exec.end_state.is_some() {
                metrics.initial_exec_end_state_hits += 1;
            }
        }
        if let Some(end_state) = exec.end_state {
            state_map.insert(tokenizer_state, end_state);
        }
        for matched in &exec.matches {
            if Some(matched.id) == ignore_terminal {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.initial_ignored_matches += 1;
                }
                continue;
            }
            // TODO: expand via mutually_greedy_group() once greedy groups
            // are wired into glrmask (see sep1 compute_commit_maps).
            let inserted = terminals_map
                .entry(tokenizer_state)
                .or_default()
                .insert(matched.id);
            if inserted {
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.initial_terminals_total += 1;
                }
            }
        }
        initial_exec_results.insert(tokenizer_state, exec);
    }

    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.initial_terminals_map_entries = terminals_map.len();
        metrics.remapped_state_entries = state_map.len();
    }

    for parser_state in state.values_mut() {
        let t_prune = metrics
            .as_ref()
            .map(|_| std::time::Instant::now());
        let mut gss = parser_state.apply_and_prune(|terminals_disallowed: &TerminalsDisallowed| {
            for (state_id, matched_terminals) in &terminals_map {
                if let Some(disallowed) = terminals_disallowed.get(state_id) {
                    if !matched_terminals.is_empty()
                        && matched_terminals
                            .iter()
                            .all(|terminal| disallowed.contains(terminal))
                    {
                        return None;
                    }
                }
            }
            Some(terminals_disallowed.clone())
        });
        if let (Some(metrics), Some(t_prune)) = (metrics.as_deref_mut(), t_prune) {
            metrics.initial_apply_prune_ns += t_prune.elapsed().as_nanos() as u64;
        }

        let t_remap = metrics
            .as_ref()
            .map(|_| std::time::Instant::now());
        gss = gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
            let mut remapped = BTreeMap::new();
            for (old_state, new_state) in &state_map {
                if let Some(disallowed) = terminals_disallowed.get(old_state) {
                    remapped
                        .entry(*new_state)
                        .or_insert_with(BTreeSet::new)
                        .extend(disallowed.iter().copied());
                }
            }
            remapped
        });
        if let (Some(metrics), Some(t_remap)) = (metrics.as_deref_mut(), t_remap) {
            metrics.initial_remap_ns += t_remap.elapsed().as_nanos() as u64;
            if gss.is_empty() {
                metrics.parser_states_pruned += 1;
            }
        }
        *parser_state = gss;
    }

    state.retain(|_, parser_state| !parser_state.is_empty());
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.parser_states_retained_after_prune = state.len();
    }

    let mut new_overall_state: BTreeMap<u32, ParserGSS> = BTreeMap::new();
    let mut processing_queue: BTreeMap<usize, BTreeMap<u32, ParserGSS>> = BTreeMap::new();

    // Take ownership instead of cloning — state will be fully replaced below.
    processing_queue.insert(0, std::mem::take(state));
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.queue_max_offsets_pending = processing_queue.len();
    }

    while let Some((offset, states_to_process)) = processing_queue.pop_first() {
        if let Some(metrics) = metrics.as_deref_mut() {
            metrics.queue_offsets_processed += 1;
            metrics.queue_states_processed += states_to_process.len();
            metrics.queue_max_states_in_offset_bucket = metrics
                .queue_max_states_in_offset_bucket
                .max(states_to_process.len());
            metrics.queue_max_offsets_pending = metrics
                .queue_max_offsets_pending
                .max(processing_queue.len() + 1);
        }

        for (tokenizer_state, gss_at_offset) in states_to_process {
            let actionable_terminals = {
                let mut terminals = BTreeSet::new();
                for state_id in gss_at_offset.peek_values() {
                    if let Some(by_terminal) = constraint.table.action.get(state_id as usize) {
                        terminals.extend(by_terminal.keys().copied());
                    }
                }
                terminals
            };
            let t_exec = metrics
                .as_ref()
                .map(|_| std::time::Instant::now());
            let (exec_result, reused_initial_exec_result) = if offset == 0 {
                match initial_exec_results.remove(&tokenizer_state) {
                    Some(exec) => (exec, true),
                    None => (
                        constraint
                            .tokenizer
                            .execute_from_state(&bytes[offset..], tokenizer_state),
                        false,
                    ),
                }
            } else {
                (
                    constraint
                        .tokenizer
                        .execute_from_state(&bytes[offset..], tokenizer_state),
                    false,
                )
            };
            if let (Some(metrics), Some(t_exec)) = (metrics.as_deref_mut(), t_exec) {
                metrics.processing_exec_calls += 1;
                metrics.processing_tokenizer_exec_ns += t_exec.elapsed().as_nanos() as u64;
                if reused_initial_exec_result {
                    metrics.reused_initial_exec_results += 1;
                }
            };

            let mut seen_matches = BTreeSet::new();
            let mut advance_cache = BTreeMap::<u32, ParserGSS>::new();
            for matched in &exec_result.matches {
                if Some(matched.id) != ignore_terminal
                    && !actionable_terminals.contains(&matched.id)
                {
                    continue;
                }
                if !seen_matches.insert((matched.width, matched.id)) {
                    continue;
                }
                if let Some(metrics) = metrics.as_deref_mut() {
                    metrics.processing_matches_total += 1;
                }
                let new_offset = offset + matched.width;

                if Some(matched.id) == ignore_terminal {
                    let next_tsid = constraint.tokenizer.initial_state();
                    if new_offset == bytes.len() {
                        let t_merge = metrics
                            .as_ref()
                            .map(|_| std::time::Instant::now());
                        let existed = new_overall_state.contains_key(&next_tsid);
                        new_overall_state
                            .entry(next_tsid)
                            .and_modify(|existing| *existing = existing.merge(&gss_at_offset))
                            .or_insert_with(|| gss_at_offset.clone());
                        if let (Some(metrics), Some(t_merge)) = (metrics.as_deref_mut(), t_merge) {
                            metrics.processing_ignored_matches += 1;
                            metrics.merge_ns += t_merge.elapsed().as_nanos() as u64;
                            if existed {
                                metrics.ignored_terminal_final_merges += 1;
                            } else {
                                metrics.ignored_terminal_final_pushes += 1;
                            }
                        }
                    } else {
                        let t_merge = metrics
                            .as_ref()
                            .map(|_| std::time::Instant::now());
                        let existed = processing_queue
                            .get(&new_offset)
                            .and_then(|states| states.get(&next_tsid))
                            .is_some();
                        processing_queue
                            .entry(new_offset)
                            .or_default()
                            .entry(next_tsid)
                            .and_modify(|existing| *existing = existing.merge(&gss_at_offset))
                            .or_insert_with(|| gss_at_offset.clone());
                        if let (Some(metrics), Some(t_merge)) = (metrics.as_deref_mut(), t_merge) {
                            metrics.processing_ignored_matches += 1;
                            metrics.merge_ns += t_merge.elapsed().as_nanos() as u64;
                            metrics.queue_max_offsets_pending = metrics
                                .queue_max_offsets_pending
                                .max(processing_queue.len());
                            if existed {
                                metrics.ignored_terminal_queue_merges += 1;
                            } else {
                                metrics.ignored_terminal_queue_pushes += 1;
                            }
                        }
                    }
                    continue;
                }

                let mut gss = if let Some(cached) = advance_cache.get(&matched.id) {
                    cached.clone()
                } else {
                    if !stack_may_advance_on(&constraint.table, &gss_at_offset, matched.id) {
                        advance_cache.insert(matched.id, ParserGSS::empty());
                        continue;
                    }
                    if let Some(metrics) = metrics.as_deref_mut() {
                        metrics.advance_stacks_calls += 1;
                    }
                    let t_advance = metrics
                        .as_ref()
                        .map(|_| std::time::Instant::now());
                    let mut advance_metrics = AdvanceStacksDebugMetrics::default();
                    let gss = advance_stacks_with_metrics(
                        &constraint.table,
                        &gss_at_offset,
                        matched.id,
                        metrics.as_deref_mut().map(|_| &mut advance_metrics),
                    );
                    if let (Some(metrics), Some(t_advance)) = (metrics.as_deref_mut(), t_advance) {
                        metrics.advance_stacks_ns += t_advance.elapsed().as_nanos() as u64;
                        if !gss.is_empty() {
                            metrics.advance_stacks_nonempty += 1;
                        }
                        accumulate_advance_stacks_metrics(metrics, &advance_metrics);
                    }
                    advance_cache.insert(matched.id, gss.clone());
                    gss
                };
                if gss.is_empty() {
                    continue;
                }

                if let Some(end_state) = exec_result.end_state {
                    if let Some(metrics) = metrics.as_deref_mut() {
                        metrics.future_group_checks += 1;
                    }
                    if constraint
                        .tokenizer
                        .dfa
                        .possible_future_group_ids(end_state)
                        .contains(matched.id as usize)
                    {
                        let t_future = metrics
                            .as_ref()
                            .map(|_| std::time::Instant::now());
                        gss = gss.apply(|terminals_disallowed: &TerminalsDisallowed| {
                            let mut updated = terminals_disallowed.clone();
                            updated.entry(end_state).or_default().insert(matched.id);
                            updated
                        });
                        if let (Some(metrics), Some(t_future)) = (metrics.as_deref_mut(), t_future) {
                            metrics.future_group_hits += 1;
                            metrics.future_group_updates += 1;
                            metrics.future_group_apply_ns += t_future.elapsed().as_nanos() as u64;
                        }
                    }
                }

                if gss.is_empty() {
                    continue;
                }

                let next_tsid = constraint.tokenizer.initial_state();
                if new_offset == bytes.len() {
                    let t_merge = metrics
                        .as_ref()
                        .map(|_| std::time::Instant::now());
                    let existed = new_overall_state.contains_key(&next_tsid);
                    new_overall_state
                        .entry(next_tsid)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert(gss);
                    if let (Some(metrics), Some(t_merge)) = (metrics.as_deref_mut(), t_merge) {
                        metrics.merge_ns += t_merge.elapsed().as_nanos() as u64;
                        if existed {
                            metrics.parser_final_merges += 1;
                        } else {
                            metrics.parser_final_pushes += 1;
                        }
                    }
                } else {
                    let t_merge = metrics
                        .as_ref()
                        .map(|_| std::time::Instant::now());
                    let existed = processing_queue
                        .get(&new_offset)
                        .and_then(|states| states.get(&next_tsid))
                        .is_some();
                    processing_queue
                        .entry(new_offset)
                        .or_default()
                        .entry(next_tsid)
                        .and_modify(|existing| *existing = existing.merge(&gss))
                        .or_insert(gss);
                    if let (Some(metrics), Some(t_merge)) = (metrics.as_deref_mut(), t_merge) {
                        metrics.merge_ns += t_merge.elapsed().as_nanos() as u64;
                        metrics.queue_max_offsets_pending = metrics
                            .queue_max_offsets_pending
                            .max(processing_queue.len());
                        if existed {
                            metrics.parser_queue_merges += 1;
                        } else {
                            metrics.parser_queue_pushes += 1;
                        }
                    }
                }
            }

            if let Some(end_state) = exec_result.end_state {
                let t_merge = metrics
                    .as_ref()
                    .map(|_| std::time::Instant::now());
                let existed = new_overall_state.contains_key(&end_state);
                new_overall_state
                    .entry(end_state)
                    .and_modify(|existing| *existing = existing.merge(&gss_at_offset))
                    .or_insert(gss_at_offset);
                if let (Some(metrics), Some(t_merge)) = (metrics.as_deref_mut(), t_merge) {
                    metrics.merge_ns += t_merge.elapsed().as_nanos() as u64;
                    if existed {
                        metrics.passthrough_end_state_merges += 1;
                    } else {
                        metrics.passthrough_end_state_pushes += 1;
                    }
                }
            }
        }
    }

    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.fused_parser_states = new_overall_state.len();
    }
    for parser_state in new_overall_state.values_mut() {
        let t_fuse = metrics
            .as_ref()
            .map(|_| std::time::Instant::now());
        *parser_state = parser_state.fuse(Some(1));
        if let (Some(metrics), Some(t_fuse)) = (metrics.as_deref_mut(), t_fuse) {
            metrics.fuse_ns += t_fuse.elapsed().as_nanos() as u64;
        }
    }
    new_overall_state.retain(|_, parser_state| !parser_state.is_empty());
    *state = new_overall_state;
    if let Some(metrics) = metrics.as_deref_mut() {
        metrics.state_summary_after = summarize_state_map(state);
        if let Some(t_total) = t_total {
            metrics.total_ns = t_total.elapsed().as_nanos() as u64;
        }
    }
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
        let bytes = self.constraint.token_bytes
            .get(&token_id)
            .ok_or_else(|| {
                format!("commit_token: token_id {token_id} not in vocabulary")
            })?;
        commit_bytes_impl(self.constraint, &mut self.state, bytes, None);
        Ok(())
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) {
        commit_bytes_impl(self.constraint, &mut self.state, bytes, None);
    }

    pub fn debug_commit_bytes_metrics(&self, bytes: &[u8]) -> CommitDebugMetrics {
        let mut cloned_state = self.state.clone();
        let mut metrics = CommitDebugMetrics {
            bytes_len: bytes.len(),
            state_summary_before: summarize_state_map(&cloned_state),
            state_summary_after: summarize_state_map(&cloned_state),
            ..CommitDebugMetrics::default()
        };
        commit_bytes_impl(self.constraint, &mut cloned_state, bytes, Some(&mut metrics));
        if bytes.is_empty() {
            metrics.state_summary_after = metrics.state_summary_before;
        }
        metrics
    }

    pub fn debug_commit_token_metrics(
        &self,
        token_id: u32,
    ) -> Result<CommitDebugMetrics, String> {
        let bytes = self.constraint.token_bytes
            .get(&token_id)
            .ok_or_else(|| {
                format!("debug_commit_token_metrics: token_id {token_id} not in vocabulary")
            })?;
        Ok(self.debug_commit_bytes_metrics(bytes))
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token(token)?;
        }
        Ok(())
    }

    pub(crate) fn process_bytes_raw(&mut self, bytes: &[u8]) {
        self.commit_bytes(bytes)
    }
}
