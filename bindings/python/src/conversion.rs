// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dict_to_vocab(token_to_id: &Bound<'_, PyDict>) -> PyResult<glrmask::Vocab> {
    let mut entries = Vec::with_capacity(token_to_id.len());
    for (key, value) in token_to_id.iter() {
        let token_bytes = key
            .downcast::<PyBytes>()
            .map_err(|_| PyValueError::new_err("vocab keys must be Python bytes"))?
            .as_bytes()
            .to_vec();
        let token_id: u32 = value.extract()?;
        entries.push((token_id, token_bytes));
    }
    Ok(glrmask::Vocab::new(entries, None))
}

fn id_to_bytes_dict_to_vocab(id_to_bytes: &Bound<'_, PyDict>) -> PyResult<glrmask::Vocab> {
    let mut entries = Vec::with_capacity(id_to_bytes.len());
    for (key, value) in id_to_bytes.iter() {
        let token_id: u32 = key.extract()?;
        let token_bytes = value
            .downcast::<PyBytes>()
            .map_err(|_| PyValueError::new_err("vocab values must be Python bytes"))?
            .as_bytes()
            .to_vec();
        entries.push((token_id, token_bytes));
    }
    Ok(glrmask::Vocab::new(entries, None))
}

fn constraint_result<T>(result: glrmask::Result<T>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("{e}")))
}

fn set_gss_summary_fields(
    dict: &Bound<'_, PyDict>,
    prefix: &str,
    path_count: usize,
    summary: &glrmask::GssProfileSummary,
) -> PyResult<()> {
    dict.set_item(format!("{prefix}_path_count"), path_count)?;
    dict.set_item(format!("{prefix}_top_values_count"), summary.top_values_count)?;
    dict.set_item(format!("{prefix}_upper_branch_nodes"), summary.upperbranch_nodes)?;
    dict.set_item(format!("{prefix}_upper_interface_nodes"), summary.interface_nodes)?;
    dict.set_item(format!("{prefix}_lower_nodes"), summary.lower_nodes)?;
    dict.set_item(
        format!("{prefix}_lower_general_nodes"),
        summary.lower_general_nodes,
    )?;
    dict.set_item(
        format!("{prefix}_lower_segment_nodes"),
        summary.lower_segment_nodes,
    )?;
    dict.set_item(
        format!("{prefix}_total_unique_nodes"),
        summary.total_unique_nodes,
    )?;
    dict.set_item(format!("{prefix}_total_edges"), summary.total_edges)?;
    dict.set_item(
        format!("{prefix}_accumulator_instances"),
        summary.accumulator_instances,
    )?;
    dict.set_item(format!("{prefix}_max_depth"), summary.max_depth)?;
    Ok(())
}

fn mask_profile_to_dict<'py>(
    py: Python<'py>,
    profile: glrmask::MaskProfile,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("total_ns", profile.total_ns)?;
    dict.set_item("cache_hit", profile.cache_hit)?;
    dict.set_item("single_path_direct", profile.single_path_direct)?;
    dict.set_item("seed_decompose_ns", profile.seed_decompose_ns)?;
    dict.set_item("queue_pop_ns", profile.queue_pop_ns)?;
    dict.set_item("loop_decompose_ns", profile.loop_decompose_ns)?;
    dict.set_item("loop_decompose_callback_ns", profile.loop_decompose_callback_ns)?;
    dict.set_item("transition_lookup_ns", profile.transition_lookup_ns)?;
    dict.set_item("transition_apply_ns", profile.transition_apply_ns)?;
    dict.set_item(
        "transition_apply_intersect_ns",
        profile.transition_apply_intersect_ns,
    )?;
    dict.set_item("transition_apply_gss_ns", profile.transition_apply_gss_ns)?;
    dict.set_item("token_accumulation_ns", profile.token_accumulation_ns)?;
    dict.set_item("enqueue_merge_ns", profile.enqueue_merge_ns)?;
    dict.set_item("queue_lookup_ns", profile.queue_lookup_ns)?;
    dict.set_item("queue_merge_ns", profile.queue_merge_ns)?;
    dict.set_item("queue_insert_ns", profile.queue_insert_ns)?;
    dict.set_item("queue_fuse_ns", profile.queue_fuse_ns)?;
    dict.set_item("finalize_ns", profile.finalize_ns)?;
    dict.set_item("finalize_zero_ns", profile.finalize_zero_ns)?;
    dict.set_item("finalize_dense_to_buf_ns", profile.finalize_dense_to_buf_ns)?;
    dict.set_item("finalize_eos_ns", profile.finalize_eos_ns)?;
    dict.set_item("finalize_cache_ns", profile.finalize_cache_ns)?;
    dict.set_item("delta_prev_available", profile.delta_prev_available)?;
    dict.set_item("delta_added_bits", profile.delta_added_bits)?;
    dict.set_item("delta_removed_bits", profile.delta_removed_bits)?;
    dict.set_item("delta_unchanged_words", profile.delta_unchanged_words)?;
    dict.set_item("delta_unchanged_bits", profile.delta_unchanged_bits)?;
    dict.set_item("delta_added_cost", profile.delta_added_cost)?;
    dict.set_item("delta_removed_cost", profile.delta_removed_cost)?;
    dict.set_item("delta_copy_cost_words", profile.delta_copy_cost_words)?;
    dict.set_item(
        "delta_scratch_estimated_cost",
        profile.delta_scratch_estimated_cost,
    )?;
    dict.set_item("delta_estimated_cost", profile.delta_estimated_cost)?;
    dict.set_item("delta_estimated_savings", profile.delta_estimated_savings)?;
    dict.set_item("delta_used_seed", profile.delta_used_seed)?;
    dict.set_item(
        "delta_added_word_group_hits",
        profile.delta_added_word_group_hits,
    )?;
    dict.set_item(
        "delta_added_word_group_entries",
        profile.delta_added_word_group_entries,
    )?;
    dict.set_item(
        "delta_removed_word_group_hits",
        profile.delta_removed_word_group_hits,
    )?;
    dict.set_item(
        "delta_removed_word_group_entries",
        profile.delta_removed_word_group_entries,
    )?;
    dict.set_item(
        "delta_added_byte_group_hits",
        profile.delta_added_byte_group_hits,
    )?;
    dict.set_item(
        "delta_added_byte_group_entries",
        profile.delta_added_byte_group_entries,
    )?;
    dict.set_item(
        "delta_removed_byte_group_hits",
        profile.delta_removed_byte_group_hits,
    )?;
    dict.set_item(
        "delta_removed_byte_group_entries",
        profile.delta_removed_byte_group_entries,
    )?;
    dict.set_item(
        "delta_added_token_iterations",
        profile.delta_added_token_iterations,
    )?;
    dict.set_item("delta_added_token_entries", profile.delta_added_token_entries)?;
    dict.set_item(
        "delta_removed_token_iterations",
        profile.delta_removed_token_iterations,
    )?;
    dict.set_item(
        "delta_removed_token_entries",
        profile.delta_removed_token_entries,
    )?;
    dict.set_item(
        "finalize_equal_dense_copy_seed",
        profile.finalize_equal_dense_copy_seed,
    )?;
    dict.set_item("finalize_delta_replay", profile.finalize_delta_replay)?;
    dict.set_item("finalize_scratch_rebuild", profile.finalize_scratch_rebuild)?;
    dict.set_item("dense_words_visited", profile.dense_words_visited)?;
    dict.set_item(
        "dense_complement_path_used",
        profile.dense_complement_path_used,
    )?;
    dict.set_item(
        "dense_normal_full_word_hits",
        profile.dense_normal_full_word_hits,
    )?;
    dict.set_item(
        "dense_normal_group_complement_hits",
        profile.dense_normal_group_complement_hits,
    )?;
    dict.set_item(
        "dense_complement_full_word_hits",
        profile.dense_complement_full_word_hits,
    )?;
    dict.set_item(
        "dense_complement_full_byte_groups",
        profile.dense_complement_full_byte_groups,
    )?;
    dict.set_item(
        "dense_complement_full_nibble_groups",
        profile.dense_complement_full_nibble_groups,
    )?;
    dict.set_item(
        "dense_complement_remaining_bits",
        profile.dense_complement_remaining_bits,
    )?;
    dict.set_item(
        "dense_normal_token_iterations",
        profile.dense_normal_token_iterations,
    )?;
    dict.set_item(
        "dense_complement_token_iterations",
        profile.dense_complement_token_iterations,
    )?;
    dict.set_item(
        "dense_normal_sparse_entries",
        profile.dense_normal_sparse_entries,
    )?;
    dict.set_item(
        "dense_normal_group_complement_sparse_entries",
        profile.dense_normal_group_complement_sparse_entries,
    )?;
    dict.set_item(
        "dense_complement_sparse_entries",
        profile.dense_complement_sparse_entries,
    )?;
    dict.set_item(
        "dense_complement_heavy_dense_clears",
        profile.dense_complement_heavy_dense_clears,
    )?;
    dict.set_item(
        "dense_complement_max_sparse_span",
        profile.dense_complement_max_sparse_span,
    )?;
    dict.set_item("dense_group_or_sparse_entries", profile.dense_group_or_sparse_entries)?;
    dict.set_item(
        "dense_group_andnot_sparse_entries",
        profile.dense_group_andnot_sparse_entries,
    )?;
    dict.set_item("enqueue_calls", profile.enqueue_calls)?;
    dict.set_item("merge_hits", profile.merge_hits)?;
    dict.set_item(
        "insert_without_merge_count",
        profile.insert_without_merge_count,
    )?;
    dict.set_item("fuse_calls", profile.fuse_calls)?;
    dict.set_item("fuse_changed_depth", profile.fuse_changed_depth)?;
    dict.set_item("stale_schedule_skips", profile.stale_schedule_skips)?;
    dict.set_item("popped_items", profile.popped_items)?;
    dict.set_item("seed_decompose_callbacks", profile.seed_decompose_callbacks)?;
    dict.set_item("loop_decompose_callbacks", profile.loop_decompose_callbacks)?;
    dict.set_item(
        "parser_dwa_transitions_enqueued",
        profile.parser_dwa_transitions_enqueued,
    )?;
    dict.set_item("other_ns", profile.other_ns)?;
    Ok(dict)
}
fn string_result<T>(result: Result<T, String>) -> PyResult<T> {
    result.map_err(PyValueError::new_err)
}

fn advance_trace_to_dict<'py>(
    py: Python<'py>,
    trace: &glrmask::AdvanceTrace,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    let det_steps = pyo3::types::PyList::empty(py);
    for step in &trace.det_steps {
        det_steps.append(advance_trace_step_to_dict(py, step)?)?;
    }
    dict.set_item("det_steps", det_steps)?;

    let nondet_waves = pyo3::types::PyList::empty(py);
    for wave in &trace.nondet_waves {
        let wave_dict = PyDict::new(py);
        wave_dict.set_item("wave_index", wave.wave_index)?;
        wave_dict.set_item("frontier_states", wave.frontier_states.clone())?;
        let branches = pyo3::types::PyList::empty(py);
        for branch in &wave.branches {
            branches.append(advance_trace_step_to_dict(py, branch)?)?;
        }
        wave_dict.set_item("branches", branches)?;
        nondet_waves.append(wave_dict)?;
    }
    dict.set_item("nondet_waves", nondet_waves)?;

    Ok(dict)
}

fn advance_trace_step_to_dict<'py>(
    py: Python<'py>,
    step: &glrmask::AdvanceTraceStep,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("source_state", step.source_state)?;
    dict.set_item("action_kind", step.action_kind.as_str())?;
    if let Some(target) = step.shift_target {
        dict.set_item("shift_target", target)?;
    }
    if let Some(replace) = step.shift_replace {
        dict.set_item("shift_replace", replace)?;
    }
    let reduces = pyo3::types::PyList::empty(py);
    for reduce in &step.reduces {
        let reduce_dict = PyDict::new(py);
        reduce_dict.set_item("lhs_nt", reduce.lhs_nt)?;
        if let Some(lhs_name) = &reduce.lhs_name {
            reduce_dict.set_item("lhs_name", lhs_name.as_str())?;
        }
        reduce_dict.set_item("pop_len", reduce.pop_len)?;
        reduce_dict.set_item("goto_sources", reduce.goto_sources.clone())?;
        let goto_targets = pyo3::types::PyList::empty(py);
        for goto in &reduce.goto_targets {
            let goto_dict = PyDict::new(py);
            goto_dict.set_item("source_state", goto.source_state)?;
            goto_dict.set_item("target_state", goto.target_state)?;
            goto_dict.set_item("replace", goto.replace)?;
            goto_targets.append(goto_dict)?;
        }
        reduce_dict.set_item("goto_targets", goto_targets)?;
        reduces.append(reduce_dict)?;
    }
    dict.set_item("reduces", reduces)?;
    Ok(dict)
}
