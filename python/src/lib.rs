//! PyO3 Python bindings for glrmask.
//!
//! Exposes `Constraint` and `ConstraintState` to Python, matching the interface
//! expected by the CFA (constraint-framework-analysis) benchmarking harness.
//!
//! # Lifetime handling
//!
//! `glrmask::ConstraintState<'a>` borrows `&'a Constraint`. PyO3 pyclass structs
//! must be `'static`, so we cannot store a `ConstraintState<'_>` directly.
//!
//! Solution: pair the `ConstraintState<'a>` with its `Arc<Constraint>` owner inside
//! a [`self_cell::self_cell!`] struct (`OwnedState`). `self_cell` generates the
//! necessary unsafe bookkeeping internally (owner outlives dependent, stable
//! address via heap allocation) and exposes a fully safe public API. No handwritten
//! `unsafe` blocks appear in this file.

use numpy::{PyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use self_cell::self_cell;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// OwnedState — `self_cell`-generated safe owner/dependent pair.
// ---------------------------------------------------------------------------

type ConstraintState<'a> = glrmask::ConstraintState<'a>;

self_cell!(
    struct OwnedState {
        owner: Arc<glrmask::Constraint>,
        #[not_covariant]
        dependent: ConstraintState,
    }
);

impl OwnedState {
    fn from_arc(arc: Arc<glrmask::Constraint>) -> Self {
        OwnedState::new(arc, |arc_ref| arc_ref.start())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dict_to_vocab(token_to_id: &Bound<'_, PyDict>) -> PyResult<glrmask::Vocab> {
    let mut entries = Vec::new();
    for (key, value) in token_to_id.iter() {
        let token_bytes: Vec<u8> = key.extract()?;
        let token_id: u32 = value.extract()?;
        entries.push((token_id, token_bytes));
    }
    Ok(glrmask::Vocab::new(entries, None))
}

fn state_summary_to_dict<'py>(
    py: Python<'py>,
    summary: glrmask::ConstraintStateSummary,
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
    metrics: glrmask::CommitMetrics,
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
        "advance_input_single_path_calls",
        metrics.advance_input_single_path_calls,
    )?;
    out.set_item(
        "advance_output_single_path_calls",
        metrics.advance_output_single_path_calls,
    )?;
    out.set_item(
        "advance_input_path_count_at_most_two_max",
        metrics.advance_input_path_count_at_most_two_max,
    )?;
    out.set_item(
        "advance_output_path_count_at_most_two_max",
        metrics.advance_output_path_count_at_most_two_max,
    )?;
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
        "advance_subtree_isolate_ns",
        metrics.advance_subtree_isolate_ns,
    )?;
    out.set_item(
        "advance_pop_cache_build_ns",
        metrics.advance_pop_cache_build_ns,
    )?;
    out.set_item(
        "advance_base_isolate_ns",
        metrics.advance_base_isolate_ns,
    )?;
    out.set_item(
        "advance_absorb_push_ns",
        metrics.advance_absorb_push_ns,
    )?;
    out.set_item(
        "advance_shift_top_values_ns",
        metrics.advance_shift_top_values_ns,
    )?;
    out.set_item(
        "advance_bookkeeping_ns",
        metrics.advance_bookkeeping_ns,
    )?;
    out.set_item(
        "advance_wrapper_ns",
        metrics.advance_wrapper_ns,
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
        "parser_queue_target_counts",
        counts_to_dict(py, &metrics.parser_queue_target_counts)?,
    )?;
    out.set_item(
        "parser_final_target_counts",
        counts_to_dict(py, &metrics.parser_final_target_counts)?,
    )?;
    out.set_item(
        "passthrough_end_state_pushes",
        metrics.passthrough_end_state_pushes,
    )?;
    out.set_item(
        "passthrough_end_state_merges",
        metrics.passthrough_end_state_merges,
    )?;
    out.set_item(
        "passthrough_end_state_counts",
        counts_to_dict(py, &metrics.passthrough_end_state_counts)?,
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
    out.set_item("bookkeeping_ns", metrics.bookkeeping_ns)?;
    out.set_item("total_ns", metrics.total_ns)?;
    Ok(out)
}

fn commit_trace_to_dict<'py>(
    py: Python<'py>,
    trace: glrmask::CommitTrace,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let exec_calls = PyList::empty(py);
    for exec_call in trace.exec_calls {
        let exec_dict = PyDict::new(py);
        exec_dict.set_item("phase", exec_call.phase)?;
        exec_dict.set_item("offset", exec_call.offset)?;
        exec_dict.set_item("start_state", exec_call.start_state)?;
        exec_dict.set_item("reused_initial_exec_result", exec_call.reused_initial_exec_result)?;
        exec_dict.set_item("end_state", exec_call.end_state)?;
        let matches = PyList::empty(py);
        for match_trace in exec_call.matches {
            let match_dict = PyDict::new(py);
            match_dict.set_item("id", match_trace.id)?;
            match_dict.set_item("width", match_trace.width)?;
            match_dict.set_item("end_state", match_trace.end_state)?;
            match_dict.set_item("ignored", match_trace.ignored)?;
            match_dict.set_item("actionable", match_trace.actionable)?;
            match_dict.set_item("advance_attempted", match_trace.advance_attempted)?;
            match_dict.set_item("advance_nonempty", match_trace.advance_nonempty)?;
            match_dict.set_item("new_offset", match_trace.new_offset)?;
            match_dict.set_item("next_tokenizer_state", match_trace.next_tokenizer_state)?;
            matches.append(match_dict)?;
        }
        exec_dict.set_item("matches", matches)?;
        exec_calls.append(exec_dict)?;
    }
    out.set_item("exec_calls", exec_calls)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// PyVocab
// ---------------------------------------------------------------------------

#[pyclass(name = "Vocab")]
#[derive(Clone)]
pub struct PyVocab {
    inner: glrmask::Vocab,
}

#[pymethods]
impl PyVocab {
    #[staticmethod]
    fn from_dict(token_to_id: &Bound<'_, PyDict>) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id)?;
        Ok(Self { inner: vocab })
    }
}

// ---------------------------------------------------------------------------
// PyConstraint
// ---------------------------------------------------------------------------

/// Compiled grammar constraint. Immutable, thread-safe.
#[pyclass(name = "Constraint")]
#[derive(Clone)]
pub struct PyConstraint {
    inner: Arc<glrmask::Constraint>,
    max_token: u32,
}

#[pymethods]
impl PyConstraint {
    #[staticmethod]
    fn from_json_schema(schema: &str, vocab: &PyVocab) -> PyResult<Self> {
        let constraint = glrmask::Constraint::from_json_schema(schema, &vocab.inner)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: vocab.inner.max_token_id(),
        })
    }

    #[staticmethod]
    fn from_lark(lark_source: &str, vocab: &PyVocab) -> PyResult<Self> {
        let constraint = glrmask::Constraint::from_lark(lark_source, &vocab.inner)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: vocab.inner.max_token_id(),
        })
    }

    #[staticmethod]
    fn from_ebnf(ebnf_source: &str, vocab: &PyVocab) -> PyResult<Self> {
        let constraint = glrmask::Constraint::from_ebnf(ebnf_source, &vocab.inner)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: vocab.inner.max_token_id(),
        })
    }

    fn save(&self) -> Vec<u8> {
        self.inner.save()
    }

    #[staticmethod]
    fn load(data: &[u8], vocab: &PyVocab) -> PyResult<Self> {
        let constraint = glrmask::Constraint::load(data)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: vocab.inner.max_token_id(),
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

// ---------------------------------------------------------------------------
// PyConstraintState
// ---------------------------------------------------------------------------

/// Mutable per-sequence parse state.
#[pyclass(name = "ConstraintState")]
pub struct PyConstraintState {
    inner: OwnedState,
    max_token: u32,
}

#[pymethods]
impl PyConstraintState {
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
        // Safety: i32 and u32 have identical size, alignment, and bit representation.
        // fill_mask writes valid u32 bitmask values where the high bit is meaningful.
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        self.inner.with_dependent(|_owner, state| state.fill_mask(buf));
        Ok(())
    }

    fn fill_mask_timed_ns(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<u64> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        let t0 = std::time::Instant::now();
        self.inner.with_dependent(|_owner, state| state.fill_mask(buf));
        Ok(t0.elapsed().as_nanos() as u64)
    }

    fn mask_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self
            .inner
            .with_dependent(|_owner, state| state.mask_metrics());

        let state_summary = state_summary_to_dict(py, metrics.state_summary)?;

        let weight_ops = PyDict::new(py);
        let total_weight_op_calls = metrics.weight_ops.union_calls
            + metrics.weight_ops.intersection_calls
            + metrics.weight_ops.difference_calls
            + metrics.weight_ops.single_intersection_calls;
        weight_ops.set_item("total_calls", total_weight_op_calls)?;
        weight_ops.set_item("union_calls", metrics.weight_ops.union_calls)?;
        weight_ops.set_item("union_memo_hits", metrics.weight_ops.union_memo_hits)?;
        weight_ops.set_item("intersection_calls", metrics.weight_ops.intersection_calls)?;
        weight_ops.set_item("intersection_memo_hits", metrics.weight_ops.intersection_memo_hits)?;
        weight_ops.set_item("difference_calls", metrics.weight_ops.difference_calls)?;
        weight_ops.set_item("difference_memo_hits", metrics.weight_ops.difference_memo_hits)?;
        weight_ops.set_item("single_intersection_calls", metrics.weight_ops.single_intersection_calls)?;
        weight_ops.set_item(
            "single_intersection_range_overlaps",
            metrics.weight_ops.single_intersection_range_overlaps,
        )?;

        let out = PyDict::new(py);
        out.set_item("state_summary", state_summary)?;
        out.set_item("weight_ops", weight_ops)?;
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

    fn commit_token_metrics<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self
            .inner
            .with_dependent(|_owner, state| state.commit_token_metrics(token_id))
            .map_err(PyValueError::new_err)?;
        commit_metrics_to_dict(py, metrics)
    }

    fn commit_bytes_metrics<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
    ) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self
            .inner
            .with_dependent(|_owner, state| state.commit_bytes_metrics(data));
        commit_metrics_to_dict(py, metrics)
    }

    fn commit_token_trace<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let trace = self
            .inner
            .with_dependent(|_owner, state| state.commit_token_trace(token_id))
            .map_err(PyValueError::new_err)?;
        commit_trace_to_dict(py, trace)
    }

    fn commit_bytes_trace<'py>(
        &self,
        py: Python<'py>,
        data: &[u8],
    ) -> PyResult<Bound<'py, PyDict>> {
        let trace = self
            .inner
            .with_dependent(|_owner, state| state.commit_bytes_trace(data));
        commit_trace_to_dict(py, trace)
    }

    fn commit_token(&mut self, token_id: u32) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| {
                state.commit_token(token_id)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
            })
    }

    fn commit_token_timed_ns(&mut self, token_id: u32) -> PyResult<u64> {
        let t0 = std::time::Instant::now();
        self.inner.with_dependent_mut(|_owner, state| {
            state.commit_token(token_id)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
        })?;
        Ok(t0.elapsed().as_nanos() as u64)
    }

    fn commit_tokens(&mut self, token_ids: Vec<u32>) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| {
                state.commit_tokens(&token_ids)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
            })
    }

    fn commit_bytes(&mut self, data: &[u8]) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| {
                state.commit_bytes(data)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
            })
    }

    fn force(&self) -> Vec<u32> {
        self.inner.with_dependent(|_owner, state| state.force())
    }

    fn is_finished(&self) -> bool {
        self.inner.with_dependent(|_owner, state| state.is_finished())
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _glrmask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVocab>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyConstraintState>()?;
    m.add_function(wrap_pyfunction!(clear_all_weights, m)?)?;
    m.add_function(wrap_pyfunction!(clear_stale_weights, m)?)?;
    m.add_function(wrap_pyfunction!(clear_weight_op_caches, m)?)?;
    m.add_function(wrap_pyfunction!(clear_weight_caches, m)?)?;
    Ok(())
}

#[pyfunction]
fn clear_all_weights() {
    glrmask::clear_all_weights();
}

#[pyfunction]
fn clear_stale_weights() {
    glrmask::clear_stale_weights();
}

#[pyfunction]
fn clear_weight_op_caches() {
    glrmask::clear_weight_op_caches();
}

/// Clear global weight interning and thread-local op memo caches.
#[pyfunction]
fn clear_weight_caches() {
    glrmask::clear_weight_caches();
}
