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
//! address via heap allocation) and exposes a safe public API for the owner /
//! dependent relationship. The only handwritten `unsafe` in this file is the
//! NumPy `i32` to `u32` bitmask view cast used by `fill_mask`.

use numpy::{PyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
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

    #[staticmethod]
    fn from_id_to_bytes(id_to_bytes: &Bound<'_, PyDict>) -> PyResult<Self> {
        let vocab = id_to_bytes_dict_to_vocab(id_to_bytes)?;
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

impl PyConstraint {
    fn from_constraint_result(
        constraint: glrmask::Result<glrmask::Constraint>,
        vocab: &PyVocab,
    ) -> PyResult<Self> {
        let constraint = constraint_result(constraint)?;
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: vocab.inner.max_token_id(),
        })
    }
}

#[pymethods]
impl PyConstraint {
    #[staticmethod]
    fn from_json_schema(schema: &str, vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::Constraint::from_json_schema(schema, &vocab.inner),
            vocab,
        )
    }

    #[staticmethod]
    fn from_lark(lark_source: &str, vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::Constraint::from_lark(lark_source, &vocab.inner),
            vocab,
        )
    }

    #[staticmethod]
    fn from_glrm_grammar(glrm_source: &str, vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::Constraint::from_glrm_grammar(glrm_source, &vocab.inner),
            vocab,
        )
    }

    #[staticmethod]
    fn from_ebnf(ebnf_source: &str, vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(
            glrmask::Constraint::from_ebnf(ebnf_source, &vocab.inner),
            vocab,
        )
    }

    fn save(&self) -> Vec<u8> {
        self.inner.save()
    }

    #[staticmethod]
    fn load(data: &[u8], vocab: &PyVocab) -> PyResult<Self> {
        Self::from_constraint_result(glrmask::Constraint::load(data), vocab)
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

    /// Return the final constraint-internal vocab remapping used by mask materialization.
    fn mask_game_mapping(&self) -> (Vec<Vec<u32>>, Vec<u32>) {
        (
            self.inner.mask_game_internal_to_original().to_vec(),
            self.inner.mask_game_original_to_internal().to_vec(),
        )
    }

    /// Return the number of GLR parser states.
    fn num_parser_states(&self) -> u32 {
        self.inner.num_parser_states()
    }

    /// Return display names for grammar terminals by terminal id.
    fn terminal_display_names(&self) -> Vec<String> {
        self.inner.terminal_display_names().to_vec()
    }

    /// Return the display name for a grammar terminal id, if present.
    fn terminal_display_name(&self, terminal_id: u32) -> Option<String> {
        self.inner
            .terminal_display_name(terminal_id)
            .map(str::to_string)
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
        let n_full_words = n / 32;
        let remainder = n % 32;
        let mut bools = vec![false; n];
        // Expand full 32-bit words in bulk — avoids per-element division/modulo.
        for (wi, &word) in words[..n_full_words].iter().enumerate() {
            let base = wi * 32;
            let mut w = word;
            for b in bools[base..base + 32].iter_mut() {
                *b = w & 1 != 0;
                w >>= 1;
            }
        }
        if remainder > 0 && n_full_words < words.len() {
            let base = n_full_words * 32;
            let mut w = words[n_full_words];
            for b in bools[base..].iter_mut() {
                *b = w & 1 != 0;
                w >>= 1;
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
        Ok(self.inner.with_dependent(|_owner, state| state.fill_mask_timed_ns(buf)))
    }

    fn fill_mask_profiled<'py>(
        &self,
        py: Python<'py>,
        mut bitmask: PyReadwriteArray1<i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        let profile = self
            .inner
            .with_dependent(|_owner, state| state.fill_mask_profiled(buf));
        mask_profile_to_dict(py, profile)
    }

    fn mask_game_fill_mask_and_internal_ids(
        &self,
        mut bitmask: PyReadwriteArray1<i32>,
    ) -> PyResult<Vec<u32>> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        Ok(self
            .inner
            .with_dependent(|_owner, state| state.mask_game_fill_mask_and_internal_ids(buf)))
    }

    fn commit_token(&mut self, token_id: u32) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_token(token_id)))
    }

    fn commit_token_timed_ns(&mut self, token_id: u32) -> PyResult<u64> {
        self.inner.with_dependent_mut(|_owner, state| {
            state
                .commit_token_timed_ns(token_id)
                .map_err(PyValueError::new_err)
        })
    }

    fn commit_tokens(&mut self, token_ids: Vec<u32>) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_tokens(&token_ids)))
    }

    /// Like commit_token but returns profiling stats as a dict.
    fn commit_token_profiled<'py>(&mut self, py: Python<'py>, token_id: u32) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let profile = self.inner.with_dependent_mut(|_owner, state| {
            state.commit_token_profiled(token_id).map_err(|e| PyValueError::new_err(e))
        })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_ns", profile.total_ns)?;
        dict.set_item("scan_ns", profile.scan_ns)?;
        dict.set_item("prune_ns", profile.prune_ns)?;
        dict.set_item("queue_ns", profile.queue_ns)?;
        dict.set_item("fuse_ns", profile.fuse_ns)?;
        dict.set_item("initial_exec_ns", profile.initial_exec_ns)?;
        dict.set_item("exec_ns", profile.exec_ns)?;
        dict.set_item("queue_exec_ns", profile.queue_exec_ns)?;
        dict.set_item("queue_match_ns", profile.queue_match_ns)?;
        dict.set_item("queue_enqueue_ns", profile.queue_enqueue_ns)?;
        dict.set_item("queue_bookkeeping_ns", profile.queue_bookkeeping_ns)?;
        dict.set_item("advance_ns", profile.advance_ns)?;
        dict.set_item("advance_may_check_ns", profile.advance_may_check_ns)?;
        dict.set_item("advance_core_ns", profile.advance_core_ns)?;
        dict.set_item("advance_future_disallow_ns", profile.advance_future_disallow_ns)?;
        dict.set_item("actionable_ns", profile.actionable_ns)?;
        dict.set_item("may_advance_ns", profile.may_advance_ns)?;
        dict.set_item("n_tokenizer_states", profile.n_tokenizer_states)?;
        dict.set_item("n_queue_entries", profile.n_queue_entries)?;
        dict.set_item("n_advances", profile.n_advances)?;
        dict.set_item("adv_n_reduces_above_floor", profile.adv_n_reduces_above_floor)?;
        dict.set_item("adv_n_floor_crossings", profile.adv_n_floor_crossings)?;
        dict.set_item("adv_n_nondet_waves", profile.adv_n_nondet_waves)?;
        dict.set_item("adv_n_nondet_branches", profile.adv_n_nondet_branches)?;
        dict.set_item("adv_clone_ns", profile.adv_clone_ns)?;
        dict.set_item("adv_fast_path_ns", profile.adv_fast_path_ns)?;
        dict.set_item("adv_stack_shift_apply_ns", profile.adv_stack_shift_apply_ns)?;
        dict.set_item("adv_det_ns", profile.adv_det_ns)?;
        dict.set_item("adv_det_floor_cross_ns", profile.adv_det_floor_cross_ns)?;
        dict.set_item("adv_nondet_ns", profile.adv_nondet_ns)?;
        dict.set_item("adv_vstack_len", profile.adv_vstack_len)?;
        dict.set_item("adv_gss_depth", profile.adv_gss_depth)?;
        dict.set_item("adv_det_exit_reason", profile.adv_det_exit_reason)?;
        dict.set_item("adv_det_exit_state", profile.adv_det_exit_state)?;
        dict.set_item("adv_n_det_action_lookups", profile.adv_n_det_action_lookups)?;
        dict.set_item("adv_n_det_goto_lookups", profile.adv_n_det_goto_lookups)?;
        dict.set_item("adv_n_det_popn_ops", profile.adv_n_det_popn_ops)?;
        dict.set_item("adv_n_nondet_reduce_ops", profile.adv_n_nondet_reduce_ops)?;
        dict.set_item("adv_n_nondet_merges", profile.adv_n_nondet_merges)?;
        dict.set_item("adv_n_nondet_isolates", profile.adv_n_nondet_isolates)?;
        dict.set_item("adv_nondet_det_ns", profile.adv_nondet_det_ns)?;
        dict.set_item(
            "adv_nondet_det_floor_cross_ns",
            profile.adv_nondet_det_floor_cross_ns,
        )?;
        dict.set_item("adv_summary_ns", profile.adv_summary_ns)?;
        dict.set_item("fast_path_total_ns", profile.fast_path_total_ns)?;
        dict.set_item("fast_path_tokenizer_exec_ns", profile.fast_path_tokenizer_exec_ns)?;
        dict.set_item("fast_path_match_scan_ns", profile.fast_path_match_scan_ns)?;
        dict.set_item("fast_path_end_state_check_ns", profile.fast_path_end_state_check_ns)?;
        dict.set_item("fast_path_prune_ns", profile.fast_path_prune_ns)?;
        dict.set_item("fast_path_advance_ns", profile.fast_path_advance_ns)?;
        dict.set_item("fast_path_future_disallow_ns", profile.fast_path_future_disallow_ns)?;
        dict.set_item("fast_path_fuse_ns", profile.fast_path_fuse_ns)?;
        dict.set_item("fast_path_state_update_ns", profile.fast_path_state_update_ns)?;
        dict.set_item("failed_fast_path_probe_ns", profile.failed_fast_path_probe_ns)?;
        dict.set_item("linear_fast_path_total_ns", profile.linear_fast_path_total_ns)?;
        dict.set_item("linear_fast_path_exec_ns", profile.linear_fast_path_exec_ns)?;
        dict.set_item("linear_fast_path_match_scan_ns", profile.linear_fast_path_match_scan_ns)?;
        dict.set_item("linear_fast_path_end_state_check_ns", profile.linear_fast_path_end_state_check_ns)?;
        dict.set_item("linear_fast_path_advance_ns", profile.linear_fast_path_advance_ns)?;
        dict.set_item("linear_fast_path_action_lookup_ns", profile.linear_fast_path_action_lookup_ns)?;
        dict.set_item("linear_fast_path_carried_gate_ns", profile.linear_fast_path_carried_gate_ns)?;
        dict.set_item("linear_fast_path_materialize_ns", profile.linear_fast_path_materialize_ns)?;
        dict.set_item("linear_fast_path_apply_action_wall_ns", profile.linear_fast_path_apply_action_wall_ns)?;
        dict.set_item("linear_fast_path_profile_bookkeeping_ns", profile.linear_fast_path_profile_bookkeeping_ns)?;
        dict.set_item("linear_fast_path_future_disallow_ns", profile.linear_fast_path_future_disallow_ns)?;
        dict.set_item("linear_fast_path_fuse_ns", profile.linear_fast_path_fuse_ns)?;
        dict.set_item("linear_fast_path_eligibility_ns", profile.linear_fast_path_eligibility_ns)?;
        dict.set_item("linear_fast_path_setup_ns", profile.linear_fast_path_setup_ns)?;
        dict.set_item("linear_fast_path_state_update_ns", profile.linear_fast_path_state_update_ns)?;
        dict.set_item("linear_fast_path_steps", profile.linear_fast_path_steps)?;
        Ok(dict)
    }

    /// Return total parser GSS root count across all tokenizer states.
    fn parser_root_count(&self) -> usize {
        self.inner.with_dependent(|_owner, state| state.parser_root_count())
    }

    /// Return parser path count (capped at limit).
    fn parser_path_count(&self, limit: usize) -> usize {
        self.inner.with_dependent(|_owner, state| state.parser_path_count(limit))
    }

    /// Return all flattened parser stacks for debugging.
    fn debug_parser_stacks(&self) -> Vec<(usize, Vec<(Vec<u32>, Vec<(usize, Vec<u32>)>)>)> {
        self.inner.with_dependent(|_owner, state| state.debug_parser_stacks())
    }

    /// Per-advance profiling: returns a list of per-advance entries and final GSS stacks.
    fn commit_token_per_advance<'py>(&mut self, py: Python<'py>, token_id: u32) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let (advances, final_stacks, commit_profile) = self.inner.with_dependent_mut(|_owner, state| {
            state.commit_token_per_advance(token_id).map_err(|e| PyValueError::new_err(e))
        })?;

        let result = pyo3::types::PyDict::new(py);

        // Convert advances to list of dicts
        let advance_list = pyo3::types::PyList::empty(py);
        for entry in advances {
            let d = pyo3::types::PyDict::new(py);
            let gss_stacks_before_len = entry.gss_stacks_before.len();
            let gss_stacks_after_len = entry.gss_stacks_after.len();
            d.set_item("terminal_id", entry.terminal_id)?;
            d.set_item("tokenizer_state", entry.tokenizer_state)?;
            d.set_item("gss_stacks_before", entry.gss_stacks_before)?;
            d.set_item("gss_stacks_after", entry.gss_stacks_after)?;
            set_gss_summary_fields(&d, "gss_before", gss_stacks_before_len, &entry.gss_summary_before)?;
            set_gss_summary_fields(&d, "gss", gss_stacks_after_len, &entry.gss_summary_after)?;
            d.set_item("match_start", entry.match_start)?;
            d.set_item("match_end", entry.match_end)?;
            d.set_item("token_bound", entry.token_bound)?;
            d.set_item("match_bytes", entry.match_bytes)?;

            // Profile fields
            let p = &entry.profile;
            d.set_item("pure_shift", p.pure_shift)?;
            d.set_item("deterministic_entered", p.deterministic_entered)?;
            d.set_item("deterministic_finished", p.deterministic_finished)?;
            d.set_item("nondeterministic_entered", p.nondeterministic_entered)?;
            d.set_item("vstack_len", p.vstack_len)?;
            d.set_item("n_reduces_above_floor", p.n_reduces_above_floor)?;
            d.set_item("n_floor_crossings", p.n_floor_crossings)?;
            d.set_item("n_nondet_waves", p.n_nondet_waves)?;
            d.set_item("n_nondet_branches", p.n_nondet_branches)?;
            d.set_item("top_states", p.top_states)?;
            d.set_item("gss_depth", p.gss_depth)?;
            d.set_item("total_ns", p.total_ns)?;
            d.set_item("clone_ns", p.clone_ns)?;
            d.set_item("fast_path_ns", p.fast_path_ns)?;
            d.set_item("stack_shift_apply_ns", p.stack_shift_apply_ns)?;
            d.set_item("det_ns", p.det_ns)?;
            d.set_item("det_floor_cross_ns", p.det_floor_cross_ns)?;
            d.set_item("nondet_ns", p.nondet_ns)?;
            d.set_item("nondet_det_ns", p.nondet_det_ns)?;
            d.set_item("nondet_det_floor_cross_ns", p.nondet_det_floor_cross_ns)?;
            d.set_item("det_exit_reason", p.det_exit_reason)?;
            d.set_item("det_exit_state", p.det_exit_state)?;
            d.set_item("n_det_action_lookups", p.n_det_action_lookups)?;
            d.set_item("n_det_goto_lookups", p.n_det_goto_lookups)?;
            d.set_item("n_det_popn_ops", p.n_det_popn_ops)?;
            d.set_item("n_nondet_reduce_ops", p.n_nondet_reduce_ops)?;
            d.set_item("n_nondet_merges", p.n_nondet_merges)?;
            d.set_item("n_nondet_isolates", p.n_nondet_isolates)?;
            if let Some(trace) = &p.trace {
                d.set_item("trace", advance_trace_to_dict(py, trace)?)?;
            }
            d.set_item("summary_ns", entry.summary_ns)?;
            advance_list.append(d)?;
        }
        result.set_item("advances", advance_list)?;
        result.set_item("final_stacks", final_stacks)?;
        let commit_dict = pyo3::types::PyDict::new(py);
        commit_dict.set_item("total_ns", commit_profile.total_ns)?;
        commit_dict.set_item("scan_ns", commit_profile.scan_ns)?;
        commit_dict.set_item("prune_ns", commit_profile.prune_ns)?;
        commit_dict.set_item("queue_ns", commit_profile.queue_ns)?;
        commit_dict.set_item("fuse_ns", commit_profile.fuse_ns)?;
        commit_dict.set_item("initial_exec_ns", commit_profile.initial_exec_ns)?;
        commit_dict.set_item("exec_ns", commit_profile.exec_ns)?;
        commit_dict.set_item("queue_exec_ns", commit_profile.queue_exec_ns)?;
        commit_dict.set_item("queue_match_ns", commit_profile.queue_match_ns)?;
        commit_dict.set_item("queue_enqueue_ns", commit_profile.queue_enqueue_ns)?;
        commit_dict.set_item("queue_bookkeeping_ns", commit_profile.queue_bookkeeping_ns)?;
        commit_dict.set_item("advance_ns", commit_profile.advance_ns)?;
        commit_dict.set_item("advance_may_check_ns", commit_profile.advance_may_check_ns)?;
        commit_dict.set_item("advance_core_ns", commit_profile.advance_core_ns)?;
        commit_dict.set_item("advance_future_disallow_ns", commit_profile.advance_future_disallow_ns)?;
        commit_dict.set_item("actionable_ns", commit_profile.actionable_ns)?;
        commit_dict.set_item("may_advance_ns", commit_profile.may_advance_ns)?;
        commit_dict.set_item("n_tokenizer_states", commit_profile.n_tokenizer_states)?;
        commit_dict.set_item("n_queue_entries", commit_profile.n_queue_entries)?;
        commit_dict.set_item("n_advances", commit_profile.n_advances)?;
        commit_dict.set_item("adv_n_reduces_above_floor", commit_profile.adv_n_reduces_above_floor)?;
        commit_dict.set_item("adv_n_floor_crossings", commit_profile.adv_n_floor_crossings)?;
        commit_dict.set_item("adv_n_nondet_waves", commit_profile.adv_n_nondet_waves)?;
        commit_dict.set_item("adv_n_nondet_branches", commit_profile.adv_n_nondet_branches)?;
        commit_dict.set_item("adv_clone_ns", commit_profile.adv_clone_ns)?;
        commit_dict.set_item("adv_fast_path_ns", commit_profile.adv_fast_path_ns)?;
        commit_dict.set_item("adv_stack_shift_apply_ns", commit_profile.adv_stack_shift_apply_ns)?;
        commit_dict.set_item("adv_det_ns", commit_profile.adv_det_ns)?;
        commit_dict.set_item(
            "adv_det_floor_cross_ns",
            commit_profile.adv_det_floor_cross_ns,
        )?;
        commit_dict.set_item("adv_nondet_ns", commit_profile.adv_nondet_ns)?;
        commit_dict.set_item("adv_vstack_len", commit_profile.adv_vstack_len)?;
        commit_dict.set_item("adv_gss_depth", commit_profile.adv_gss_depth)?;
        commit_dict.set_item("adv_det_exit_reason", commit_profile.adv_det_exit_reason)?;
        commit_dict.set_item("adv_det_exit_state", commit_profile.adv_det_exit_state)?;
        commit_dict.set_item("adv_n_det_action_lookups", commit_profile.adv_n_det_action_lookups)?;
        commit_dict.set_item("adv_n_det_goto_lookups", commit_profile.adv_n_det_goto_lookups)?;
        commit_dict.set_item("adv_n_det_popn_ops", commit_profile.adv_n_det_popn_ops)?;
        commit_dict.set_item("adv_n_nondet_reduce_ops", commit_profile.adv_n_nondet_reduce_ops)?;
        commit_dict.set_item("adv_n_nondet_merges", commit_profile.adv_n_nondet_merges)?;
        commit_dict.set_item("adv_n_nondet_isolates", commit_profile.adv_n_nondet_isolates)?;
        commit_dict.set_item("adv_nondet_det_ns", commit_profile.adv_nondet_det_ns)?;
        commit_dict.set_item(
            "adv_nondet_det_floor_cross_ns",
            commit_profile.adv_nondet_det_floor_cross_ns,
        )?;
        commit_dict.set_item("adv_summary_ns", commit_profile.adv_summary_ns)?;
        commit_dict.set_item("fast_path_total_ns", commit_profile.fast_path_total_ns)?;
        commit_dict.set_item("fast_path_tokenizer_exec_ns", commit_profile.fast_path_tokenizer_exec_ns)?;
        commit_dict.set_item("fast_path_match_scan_ns", commit_profile.fast_path_match_scan_ns)?;
        commit_dict.set_item("fast_path_end_state_check_ns", commit_profile.fast_path_end_state_check_ns)?;
        commit_dict.set_item("fast_path_prune_ns", commit_profile.fast_path_prune_ns)?;
        commit_dict.set_item("fast_path_advance_ns", commit_profile.fast_path_advance_ns)?;
        commit_dict.set_item("fast_path_future_disallow_ns", commit_profile.fast_path_future_disallow_ns)?;
        commit_dict.set_item("fast_path_fuse_ns", commit_profile.fast_path_fuse_ns)?;
        commit_dict.set_item("fast_path_state_update_ns", commit_profile.fast_path_state_update_ns)?;
        commit_dict.set_item("linear_fast_path_total_ns", commit_profile.linear_fast_path_total_ns)?;
        commit_dict.set_item("linear_fast_path_exec_ns", commit_profile.linear_fast_path_exec_ns)?;
        commit_dict.set_item("linear_fast_path_match_scan_ns", commit_profile.linear_fast_path_match_scan_ns)?;
        commit_dict.set_item("linear_fast_path_end_state_check_ns", commit_profile.linear_fast_path_end_state_check_ns)?;
        commit_dict.set_item("linear_fast_path_advance_ns", commit_profile.linear_fast_path_advance_ns)?;
        commit_dict.set_item("linear_fast_path_action_lookup_ns", commit_profile.linear_fast_path_action_lookup_ns)?;
        commit_dict.set_item("linear_fast_path_carried_gate_ns", commit_profile.linear_fast_path_carried_gate_ns)?;
        commit_dict.set_item("linear_fast_path_materialize_ns", commit_profile.linear_fast_path_materialize_ns)?;
        commit_dict.set_item("linear_fast_path_apply_action_wall_ns", commit_profile.linear_fast_path_apply_action_wall_ns)?;
        commit_dict.set_item("linear_fast_path_profile_bookkeeping_ns", commit_profile.linear_fast_path_profile_bookkeeping_ns)?;
        commit_dict.set_item("linear_fast_path_future_disallow_ns", commit_profile.linear_fast_path_future_disallow_ns)?;
        commit_dict.set_item("linear_fast_path_fuse_ns", commit_profile.linear_fast_path_fuse_ns)?;
        commit_dict.set_item("linear_fast_path_steps", commit_profile.linear_fast_path_steps)?;
        result.set_item("commit_profile", commit_dict)?;

        Ok(result)
    }

    fn commit_bytes(&mut self, data: &[u8]) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_bytes(data)))
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
    m.add_function(wrap_pyfunction!(clear_stale_weights, m)?)?;
    m.add_function(wrap_pyfunction!(clear_weight_op_caches, m)?)?;
    m.add_function(wrap_pyfunction!(compile_grammar_def_json, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_grammar_glrm, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_vocab_for_compile, m)?)?;
    Ok(())
}

#[pyfunction]
fn clear_stale_weights() {
    glrmask::clear_stale_weights();
}

#[pyfunction]
fn clear_weight_op_caches() {
    glrmask::clear_weight_op_caches();
}

#[pyfunction]
fn prepare_vocab_for_compile(vocab: &PyVocab) {
    glrmask::prepare_vocab_for_compile(&vocab.inner);
}

#[pyfunction]
fn compile_grammar_def_json(grammar_def_json: &str, vocab: &PyVocab) -> PyResult<PyConstraint> {
    let constraint = glrmask::compile_grammar_def_json(grammar_def_json, &vocab.inner)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let max_token = vocab.inner.max_token_id();
    Ok(PyConstraint {
        inner: std::sync::Arc::new(constraint),
        max_token,
    })
}

#[pyfunction]
fn dump_json_schema_grammar_glrm(schema_json: &str) -> PyResult<String> {
    glrmask::dump_json_schema_grammar_glrm(schema_json)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}
