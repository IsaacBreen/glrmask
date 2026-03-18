use numpy::{PyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use self_cell::self_cell;
use std::sync::Arc;

type ConstraintState<'a> = glrmask_runtime::ConstraintState<'a>;

self_cell!(
    struct OwnedState {
        owner: Arc<glrmask_runtime::Constraint>,
        #[not_covariant]
        dependent: ConstraintState,
    }
);

impl OwnedState {
    fn from_arc(arc: Arc<glrmask_runtime::Constraint>) -> Self {
        OwnedState::new(arc, |arc_ref| arc_ref.start())
    }

    fn from_saved(arc: Arc<glrmask_runtime::Constraint>, data: &[u8]) -> PyResult<Self> {
        arc.load_state(data)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let saved = data.to_vec();
        Ok(OwnedState::new(arc, move |arc_ref| {
            arc_ref.load_state(&saved).expect("validated above")
        }))
    }
}

fn dict_to_vocab(token_to_id: &Bound<'_, PyDict>) -> PyResult<glrmask_runtime::Vocab> {
    let mut entries = Vec::new();
    for (key, value) in token_to_id.iter() {
        let token_bytes: Vec<u8> = key.extract()?;
        let token_id: u32 = value.extract()?;
        entries.push((token_id, token_bytes));
    }
    Ok(glrmask_runtime::Vocab::new(entries, None))
}

fn vocab_to_dict<'py>(py: Python<'py>, vocab: &glrmask_runtime::Vocab) -> PyResult<Bound<'py, PyDict>> {
    let token_to_id = PyDict::new(py);
    for (token_id, token_bytes) in &vocab.entries {
        token_to_id.set_item(token_bytes.as_slice(), *token_id)?;
    }
    Ok(token_to_id)
}

fn state_summary_to_dict<'py>(
    py: Python<'py>,
    summary: glrmask_runtime::ConstraintStateSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("tokenizer_state_count", summary.tokenizer_state_count)?;
    out.set_item(
        "nonempty_tokenizer_state_count",
        summary.nonempty_tokenizer_state_count,
    )?;
    out.set_item("parser_top_values_total", summary.parser_top_values_total)?;
    out.set_item("parser_top_values_max", summary.parser_top_values_max)?;
    out.set_item(
        "parser_upperbranch_nodes_total",
        summary.parser_upperbranch_nodes_total,
    )?;
    out.set_item(
        "parser_upperbranch_nodes_max",
        summary.parser_upperbranch_nodes_max,
    )?;
    out.set_item(
        "parser_interface_nodes_total",
        summary.parser_interface_nodes_total,
    )?;
    out.set_item(
        "parser_interface_nodes_max",
        summary.parser_interface_nodes_max,
    )?;
    out.set_item("parser_lower_nodes_total", summary.parser_lower_nodes_total)?;
    out.set_item("parser_lower_nodes_max", summary.parser_lower_nodes_max)?;
    out.set_item("parser_unique_nodes_total", summary.parser_unique_nodes_total)?;
    out.set_item("parser_unique_nodes_max", summary.parser_unique_nodes_max)?;
    out.set_item("parser_total_edges_total", summary.parser_total_edges_total)?;
    out.set_item(
        "parser_accumulator_instances_total",
        summary.parser_accumulator_instances_total,
    )?;
    out.set_item("parser_max_depth", summary.parser_max_depth)?;
    Ok(out)
}

fn counts_to_dict<'py, K>(
    py: Python<'py>,
    counts: &std::collections::BTreeMap<K, usize>,
) -> PyResult<Bound<'py, PyDict>>
where
    K: pyo3::IntoPyObject<'py> + Copy,
{
    let out = PyDict::new(py);
    for (&key, &value) in counts {
        out.set_item(key, value)?;
    }
    Ok(out)
}

fn commit_metrics_to_dict<'py>(
    py: Python<'py>,
    metrics: glrmask_runtime::CommitDebugMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("bytes_len", metrics.bytes_len)?;
    out.set_item(
        "state_summary_before",
        state_summary_to_dict(py, metrics.state_summary_before)?,
    )?;
    out.set_item(
        "state_summary_after",
        state_summary_to_dict(py, metrics.state_summary_after)?,
    )?;
    out.set_item("initial_tokenizer_states", metrics.initial_tokenizer_states)?;
    out.set_item("initial_exec_calls", metrics.initial_exec_calls)?;
    out.set_item("initial_exec_end_state_hits", metrics.initial_exec_end_state_hits)?;
    out.set_item("initial_matches_total", metrics.initial_matches_total)?;
    out.set_item("initial_ignored_matches", metrics.initial_ignored_matches)?;
    out.set_item("initial_terminals_total", metrics.initial_terminals_total)?;
    out.set_item("initial_terminals_map_entries", metrics.initial_terminals_map_entries)?;
    out.set_item("remapped_state_entries", metrics.remapped_state_entries)?;
    out.set_item("parser_states_pruned", metrics.parser_states_pruned)?;
    out.set_item(
        "parser_states_retained_after_prune",
        metrics.parser_states_retained_after_prune,
    )?;
    out.set_item("queue_offsets_processed", metrics.queue_offsets_processed)?;
    out.set_item("queue_states_processed", metrics.queue_states_processed)?;
    out.set_item("queue_max_offsets_pending", metrics.queue_max_offsets_pending)?;
    out.set_item(
        "queue_max_states_in_offset_bucket",
        metrics.queue_max_states_in_offset_bucket,
    )?;
    out.set_item("processing_exec_calls", metrics.processing_exec_calls)?;
    out.set_item("reused_initial_exec_results", metrics.reused_initial_exec_results)?;
    out.set_item("processing_matches_total", metrics.processing_matches_total)?;
    out.set_item("processing_ignored_matches", metrics.processing_ignored_matches)?;
    out.set_item("advance_stacks_calls", metrics.advance_stacks_calls)?;
    out.set_item("advance_stacks_nonempty", metrics.advance_stacks_nonempty)?;
    out.set_item(
        "advance_reduce_closure_iterations_total",
        metrics.advance_reduce_closure_iterations_total,
    )?;
    out.set_item(
        "advance_reduce_closure_iterations_max",
        metrics.advance_reduce_closure_iterations_max,
    )?;
    out.set_item(
        "advance_frontier_states_total",
        metrics.advance_frontier_states_total,
    )?;
    out.set_item("advance_frontier_states_max", metrics.advance_frontier_states_max)?;
    out.set_item(
        "advance_reduce_rules_considered",
        metrics.advance_reduce_rules_considered,
    )?;
    out.set_item("advance_popn_calls", metrics.advance_popn_calls)?;
    out.set_item("advance_popn_nonempty", metrics.advance_popn_nonempty)?;
    out.set_item("advance_goto_lookups", metrics.advance_goto_lookups)?;
    out.set_item("advance_goto_hits", metrics.advance_goto_hits)?;
    out.set_item(
        "advance_reductions_emitted",
        metrics.advance_reductions_emitted,
    )?;
    out.set_item("advance_absorb_targets", metrics.advance_absorb_targets)?;
    out.set_item(
        "advance_shift_state_candidates",
        metrics.advance_shift_state_candidates,
    )?;
    out.set_item("advance_shift_targets_hit", metrics.advance_shift_targets_hit)?;
    out.set_item("advance_shifted_results", metrics.advance_shifted_results)?;
    out.set_item(
        "advance_reduce_rule_considered_counts",
        counts_to_dict(py, &metrics.advance_reduce_rule_considered_counts)?,
    )?;
    out.set_item(
        "advance_reduce_rule_emitted_counts",
        counts_to_dict(py, &metrics.advance_reduce_rule_emitted_counts)?,
    )?;
    out.set_item(
        "advance_reduce_rhs_len_emitted_counts",
        counts_to_dict(py, &metrics.advance_reduce_rhs_len_emitted_counts)?,
    )?;
    out.set_item(
        "advance_reduce_lhs_emitted_counts",
        counts_to_dict(py, &metrics.advance_reduce_lhs_emitted_counts)?,
    )?;
    out.set_item(
        "advance_reduce_state_emitted_counts",
        counts_to_dict(py, &metrics.advance_reduce_state_emitted_counts)?,
    )?;
    out.set_item(
        "advance_goto_from_counts",
        counts_to_dict(py, &metrics.advance_goto_from_counts)?,
    )?;
    out.set_item(
        "advance_goto_target_counts",
        counts_to_dict(py, &metrics.advance_goto_target_counts)?,
    )?;
    out.set_item(
        "advance_input_top_values_total",
        metrics.advance_input_top_values_total,
    )?;
    out.set_item(
        "advance_input_top_values_max",
        metrics.advance_input_top_values_max,
    )?;
    out.set_item(
        "advance_input_upperbranch_nodes_total",
        metrics.advance_input_upperbranch_nodes_total,
    )?;
    out.set_item(
        "advance_input_upperbranch_nodes_max",
        metrics.advance_input_upperbranch_nodes_max,
    )?;
    out.set_item(
        "advance_input_interface_nodes_total",
        metrics.advance_input_interface_nodes_total,
    )?;
    out.set_item(
        "advance_input_interface_nodes_max",
        metrics.advance_input_interface_nodes_max,
    )?;
    out.set_item(
        "advance_input_lower_nodes_total",
        metrics.advance_input_lower_nodes_total,
    )?;
    out.set_item(
        "advance_input_lower_nodes_max",
        metrics.advance_input_lower_nodes_max,
    )?;
    out.set_item(
        "advance_input_unique_nodes_total",
        metrics.advance_input_unique_nodes_total,
    )?;
    out.set_item(
        "advance_input_unique_nodes_max",
        metrics.advance_input_unique_nodes_max,
    )?;
    out.set_item(
        "advance_input_total_edges_total",
        metrics.advance_input_total_edges_total,
    )?;
    out.set_item(
        "advance_input_total_edges_max",
        metrics.advance_input_total_edges_max,
    )?;
    out.set_item(
        "advance_output_top_values_total",
        metrics.advance_output_top_values_total,
    )?;
    out.set_item(
        "advance_output_top_values_max",
        metrics.advance_output_top_values_max,
    )?;
    out.set_item(
        "advance_output_upperbranch_nodes_total",
        metrics.advance_output_upperbranch_nodes_total,
    )?;
    out.set_item(
        "advance_output_upperbranch_nodes_max",
        metrics.advance_output_upperbranch_nodes_max,
    )?;
    out.set_item(
        "advance_output_interface_nodes_total",
        metrics.advance_output_interface_nodes_total,
    )?;
    out.set_item(
        "advance_output_interface_nodes_max",
        metrics.advance_output_interface_nodes_max,
    )?;
    out.set_item(
        "advance_output_lower_nodes_total",
        metrics.advance_output_lower_nodes_total,
    )?;
    out.set_item(
        "advance_output_lower_nodes_max",
        metrics.advance_output_lower_nodes_max,
    )?;
    out.set_item(
        "advance_output_unique_nodes_total",
        metrics.advance_output_unique_nodes_total,
    )?;
    out.set_item(
        "advance_output_unique_nodes_max",
        metrics.advance_output_unique_nodes_max,
    )?;
    out.set_item(
        "advance_output_total_edges_total",
        metrics.advance_output_total_edges_total,
    )?;
    out.set_item(
        "advance_output_total_edges_max",
        metrics.advance_output_total_edges_max,
    )?;
    out.set_item("future_group_checks", metrics.future_group_checks)?;
    out.set_item("future_group_hits", metrics.future_group_hits)?;
    out.set_item("future_group_updates", metrics.future_group_updates)?;
    out.set_item(
        "ignored_terminal_queue_pushes",
        metrics.ignored_terminal_queue_pushes,
    )?;
    out.set_item(
        "ignored_terminal_queue_merges",
        metrics.ignored_terminal_queue_merges,
    )?;
    out.set_item(
        "ignored_terminal_final_pushes",
        metrics.ignored_terminal_final_pushes,
    )?;
    out.set_item(
        "ignored_terminal_final_merges",
        metrics.ignored_terminal_final_merges,
    )?;
    out.set_item("parser_queue_pushes", metrics.parser_queue_pushes)?;
    out.set_item("parser_queue_merges", metrics.parser_queue_merges)?;
    out.set_item("parser_final_pushes", metrics.parser_final_pushes)?;
    out.set_item("parser_final_merges", metrics.parser_final_merges)?;
    out.set_item(
        "passthrough_end_state_pushes",
        metrics.passthrough_end_state_pushes,
    )?;
    out.set_item(
        "passthrough_end_state_merges",
        metrics.passthrough_end_state_merges,
    )?;
    out.set_item("fused_parser_states", metrics.fused_parser_states)?;
    out.set_item("initial_tokenizer_exec_ns", metrics.initial_tokenizer_exec_ns)?;
    out.set_item("initial_apply_prune_ns", metrics.initial_apply_prune_ns)?;
    out.set_item("initial_remap_ns", metrics.initial_remap_ns)?;
    out.set_item("processing_tokenizer_exec_ns", metrics.processing_tokenizer_exec_ns)?;
    out.set_item("advance_stacks_ns", metrics.advance_stacks_ns)?;
    out.set_item("future_group_apply_ns", metrics.future_group_apply_ns)?;
    out.set_item("merge_ns", metrics.merge_ns)?;
    out.set_item("fuse_ns", metrics.fuse_ns)?;
    out.set_item("total_ns", metrics.total_ns)?;
    Ok(out)
}

fn load_runtime_constraint(data: &[u8], vocab: &PyVocab) -> PyResult<PyConstraint> {
    let constraint = glrmask_runtime::Constraint::load(data)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(PyConstraint {
        inner: Arc::new(constraint),
        max_token: vocab.inner.max_token_id(),
    })
}

fn build_via_glrmask(py: Python<'_>, method: &str, source: &str, vocab: &PyVocab) -> PyResult<PyConstraint> {
    let glrmask = py
        .import("_glrmask")
        .map_err(|e| PyValueError::new_err(format!("glrmask_runtime requires _glrmask for {method}: {e}")))?;
    let glrmask_vocab = glrmask
        .getattr("Vocab")?
        .call_method1("from_dict", (vocab_to_dict(py, &vocab.inner)?,))?;
    let source_constraint = glrmask
        .getattr("Constraint")?
        .call_method1(method, (source, glrmask_vocab))?;
    let saved: Vec<u8> = source_constraint.call_method0("save")?.extract()?;
    load_runtime_constraint(&saved, vocab)
}

#[pyclass(name = "Vocab")]
#[derive(Clone)]
pub struct PyVocab {
    inner: glrmask_runtime::Vocab,
}

#[pymethods]
impl PyVocab {
    #[staticmethod]
    fn from_dict(token_to_id: &Bound<'_, PyDict>) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id)?;
        Ok(Self { inner: vocab })
    }
}

#[pyclass(name = "Constraint")]
#[derive(Clone)]
pub struct PyConstraint {
    inner: Arc<glrmask_runtime::Constraint>,
    max_token: u32,
}

#[pymethods]
impl PyConstraint {
    #[staticmethod]
    fn from_json_schema(py: Python<'_>, schema: &str, vocab: &PyVocab) -> PyResult<Self> {
        build_via_glrmask(py, "from_json_schema", schema, vocab)
    }

    #[staticmethod]
    fn from_lark(py: Python<'_>, lark_source: &str, vocab: &PyVocab) -> PyResult<Self> {
        build_via_glrmask(py, "from_lark", lark_source, vocab)
    }

    #[staticmethod]
    fn from_ebnf(py: Python<'_>, ebnf_source: &str, vocab: &PyVocab) -> PyResult<Self> {
        build_via_glrmask(py, "from_ebnf", ebnf_source, vocab)
    }

    fn save(&self) -> Vec<u8> {
        self.inner.save()
    }

    #[staticmethod]
    fn load(data: &[u8], vocab: &PyVocab) -> PyResult<Self> {
        load_runtime_constraint(data, vocab)
    }

    fn load_state(&self, data: &[u8]) -> PyResult<PyConstraintState> {
        Ok(PyConstraintState {
            inner: OwnedState::from_saved(self.inner.clone(), data)?,
            max_token: self.max_token,
        })
    }

    fn start(&self) -> PyConstraintState {
        PyConstraintState {
            inner: OwnedState::from_arc(self.inner.clone()),
            max_token: self.max_token,
        }
    }

    fn mask_len(&self) -> usize {
        self.inner.mask_len()
    }
}

#[pyclass(name = "ConstraintState")]
pub struct PyConstraintState {
    inner: OwnedState,
    max_token: u32,
}

#[pymethods]
impl PyConstraintState {
    fn save(&self) -> Vec<u8> {
        self.inner.with_dependent(|_owner, state| state.save())
    }

    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let words = self.inner.with_dependent(|_owner, state| state.mask());
        let n = (self.max_token + 1) as usize;
        let mut bools = vec![false; n];
        for i in 0..n {
            let (wi, bi) = (i / 32, i % 32);
            if wi < words.len() && (words[wi] >> bi) & 1 != 0 {
                bools[i] = true;
            }
        }
        Ok(PyArray1::from_vec(py, bools))
    }

    fn fill_mask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        self.inner.with_dependent(|_owner, state| state.fill_mask(buf));
        Ok(())
    }

    fn debug_mask_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self
            .inner
            .with_dependent(|_owner, state| state.debug_mask_metrics());
        let out = PyDict::new(py);
        out.set_item("state_summary", state_summary_to_dict(py, metrics.state_summary)?)?;
        out.set_item("mask_words", metrics.mask_words)?;
        out.set_item("allowed_token_count", metrics.allowed_token_count)?;
        out.set_item("seeded_entries", metrics.seeded_entries)?;
        out.set_item("seeded_empty_after_weight", metrics.seeded_empty_after_weight)?;
        out.set_item("queue_depth_buckets_processed", metrics.queue_depth_buckets_processed)?;
        out.set_item("queue_items_processed", metrics.queue_items_processed)?;
        out.set_item("final_weight_checks", metrics.final_weight_checks)?;
        out.set_item("final_weight_full_hits", metrics.final_weight_full_hits)?;
        out.set_item("final_weight_intersection_hits", metrics.final_weight_intersection_hits)?;
        out.set_item("parser_states_peeked", metrics.parser_states_peeked)?;
        out.set_item("transitions_considered", metrics.transitions_considered)?;
        out.set_item("transitions_hit", metrics.transitions_hit)?;
        out.set_item("transitions_missing", metrics.transitions_missing)?;
        out.set_item("transitions_popped_empty", metrics.transitions_popped_empty)?;
        out.set_item("transitions_pruned_empty", metrics.transitions_pruned_empty)?;
        out.set_item("transitions_enqueued", metrics.transitions_enqueued)?;
        out.set_item("max_queue_items", metrics.max_queue_items)?;
        out.set_item("max_weighted_gss_top_values", metrics.max_weighted_gss_top_values)?;
        out.set_item("max_weighted_gss_unique_nodes", metrics.max_weighted_gss_unique_nodes)?;
        out.set_item("max_weighted_gss_total_edges", metrics.max_weighted_gss_total_edges)?;
        out.set_item("max_weighted_gss_depth", metrics.max_weighted_gss_depth)?;
        out.set_item("max_depth_bucket_processed", metrics.max_depth_bucket_processed)?;
        out.set_item("min_depth_bucket_processed", metrics.min_depth_bucket_processed)?;
        out.set_item("max_items_in_depth_bucket", metrics.max_items_in_depth_bucket)?;
        out.set_item("positive_transitions_hit", metrics.positive_transitions_hit)?;
        out.set_item("positive_transitions_enqueued", metrics.positive_transitions_enqueued)?;
        out.set_item("default_transitions_hit", metrics.default_transitions_hit)?;
        out.set_item("default_transitions_enqueued", metrics.default_transitions_enqueued)?;
        out.set_item("seed_ns", metrics.seed_ns)?;
        out.set_item("final_weight_ns", metrics.final_weight_ns)?;
        out.set_item("transition_gss_ns", metrics.transition_gss_ns)?;
        out.set_item("transition_intersect_ns", metrics.transition_intersect_ns)?;
        out.set_item("transition_enqueue_ns", metrics.transition_enqueue_ns)?;
        out.set_item("queue_pop_ns", metrics.queue_pop_ns)?;
        out.set_item("bfs_loop_ns", metrics.bfs_loop_ns)?;
        out.set_item("total_ns", metrics.total_ns)?;
        out.set_item("internal_token_dense_words", metrics.internal_token_dense_words)?;
        Ok(out)
    }

    fn debug_commit_token_metrics<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self
            .inner
            .with_dependent(|_owner, state| state.debug_commit_token_metrics(token_id))
            .map_err(PyValueError::new_err)?;
        commit_metrics_to_dict(py, metrics)
    }

    fn debug_commit_bytes_metrics<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self
            .inner
            .with_dependent(|_owner, state| state.debug_commit_bytes_metrics(data));
        commit_metrics_to_dict(py, metrics)
    }

    fn commit_token(&mut self, token_id: u32) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| {
                state.commit_token(token_id)
                    .map_err(pyo3::exceptions::PyValueError::new_err)
            })
    }

    fn commit_tokens(&mut self, token_ids: Vec<u32>) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| {
                state.commit_tokens(&token_ids)
                    .map_err(pyo3::exceptions::PyValueError::new_err)
            })
    }

    fn commit_bytes(&mut self, data: &[u8]) {
        self.inner
            .with_dependent_mut(|_owner, state| state.commit_bytes(data));
    }

    fn force(&self) -> Vec<u32> {
        self.inner.with_dependent(|_owner, state| state.force())
    }

    fn is_finished(&self) -> bool {
        self.inner.with_dependent(|_owner, state| state.is_finished())
    }
}

#[pymodule]
fn _glrmask_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVocab>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyConstraintState>()?;
    Ok(())
}
