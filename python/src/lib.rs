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
use pyo3::exceptions::PyNotImplementedError;
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

fn string_result<T>(result: Result<T, String>) -> PyResult<T> {
    result.map_err(PyValueError::new_err)
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

    /// Return the number of GLR parser states.
    fn num_parser_states(&self) -> u32 {
        self.inner.num_parser_states()
    }

    /// Temporary diagnostic: return the original→internal token mapping as a list.
    fn _debug_token_map(&self) -> PyResult<Vec<u32>> {
        Err(PyNotImplementedError::new_err("_debug_token_map is not available in this build"))
    }

    /// Temporary diagnostic: return internal→originals mapping.
    fn _debug_internal_to_tokens(&self) -> PyResult<Vec<Vec<u32>>> {
        Err(PyNotImplementedError::new_err("_debug_internal_to_tokens is not available in this build"))
    }

    /// Walk bytes through DFA from every state. Returns list of (final_state, finalizers, futures).
    fn _debug_walk_dfa(&self, token_bytes: Vec<u8>) -> PyResult<Vec<(u32, Vec<u32>, Vec<u32>)>> {
        let _ = token_bytes;
        Err(PyNotImplementedError::new_err("_debug_walk_dfa is not available in this build"))
    }

    /// Return action table entries for a given parser state.
    /// Returns list of (terminal_id, action_str) pairs.
    fn _debug_actions_for_state(&self, state: u32) -> PyResult<Vec<(u32, String)>> {
        let _ = state;
        Err(PyNotImplementedError::new_err("_debug_actions_for_state is not available in this build"))
    }

    /// Return rule info: list of (lhs_nonterminal, rhs_length, rhs_symbols_debug).
    fn _debug_rules(&self) -> PyResult<Vec<(u32, usize, String)>> {
        Err(PyNotImplementedError::new_err("_debug_rules is not available in this build"))
    }

    /// Return terminal names/mapping for the GLR table.
    fn _debug_num_terminals(&self) -> PyResult<u32> {
        Err(PyNotImplementedError::new_err("_debug_num_terminals is not available in this build"))
    }

    fn _debug_num_states(&self) -> PyResult<u32> {
        Err(PyNotImplementedError::new_err("_debug_num_states is not available in this build"))
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
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let buf: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        let t = self.inner.with_dependent(|_owner, state| state.fill_mask_profiled(buf));
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("cache_hit_ns", t.cache_hit_ns)?;
        dict.set_item("cache_miss_ns", t.cache_miss_ns)?;
        dict.set_item("seed_ns", t.seed_ns)?;
        dict.set_item("bfs_ns", t.bfs_ns)?;
        dict.set_item("convert_ns", t.convert_ns)?;
        dict.set_item("total_ns", t.total_ns)?;
        dict.set_item("bfs_queue_pops", t.bfs_queue_pops)?;
        dict.set_item("bfs_states_processed", t.bfs_states_processed)?;
        dict.set_item("weight_intersections", t.weight_intersections)?;
        dict.set_item("weight_pruned", t.weight_pruned)?;
        dict.set_item("convert_incremental", t.convert_incremental)?;
        dict.set_item("convert_delta_tokens", t.convert_delta_tokens)?;
        dict.set_item("seed_tokenizer_states", t.seed_tokenizer_states)?;
        dict.set_item("seed_chain_hits", t.seed_chain_hits)?;
        dict.set_item("seed_chain_misses", t.seed_chain_misses)?;
        dict.set_item("bfs_fast_path_ns", t.bfs_fast_path_ns)?;
        dict.set_item("bfs_standard_path_ns", t.bfs_standard_path_ns)?;
        dict.set_item("bfs_fw_merge_ns", t.bfs_fw_merge_ns)?;
        Ok(dict)
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
        dict.set_item("exec_ns", profile.exec_ns)?;
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
        dict.set_item("adv_det_ns", profile.adv_det_ns)?;
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
        dict.set_item("fast_path_total_ns", profile.fast_path_total_ns)?;
        dict.set_item("fast_path_tokenizer_exec_ns", profile.fast_path_tokenizer_exec_ns)?;
        dict.set_item("fast_path_match_scan_ns", profile.fast_path_match_scan_ns)?;
        dict.set_item("fast_path_end_state_check_ns", profile.fast_path_end_state_check_ns)?;
        dict.set_item("fast_path_prune_ns", profile.fast_path_prune_ns)?;
        dict.set_item("fast_path_advance_ns", profile.fast_path_advance_ns)?;
        dict.set_item("fast_path_future_disallow_ns", profile.fast_path_future_disallow_ns)?;
        dict.set_item("fast_path_fuse_ns", profile.fast_path_fuse_ns)?;
        dict.set_item("fast_path_state_update_ns", profile.fast_path_state_update_ns)?;
        dict.set_item("linear_fast_path_total_ns", profile.linear_fast_path_total_ns)?;
        dict.set_item("linear_fast_path_exec_ns", profile.linear_fast_path_exec_ns)?;
        dict.set_item("linear_fast_path_match_scan_ns", profile.linear_fast_path_match_scan_ns)?;
        dict.set_item("linear_fast_path_end_state_check_ns", profile.linear_fast_path_end_state_check_ns)?;
        dict.set_item("linear_fast_path_advance_ns", profile.linear_fast_path_advance_ns)?;
        dict.set_item("linear_fast_path_future_disallow_ns", profile.linear_fast_path_future_disallow_ns)?;
        dict.set_item("linear_fast_path_fuse_ns", profile.linear_fast_path_fuse_ns)?;
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
    fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
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

            d.set_item("gss_before_path_count", gss_stacks_before_len)?;
            d.set_item("gss_before_top_values_count", entry.gss_summary_before.top_values_count)?;
            d.set_item("gss_before_upper_branch_nodes", entry.gss_summary_before.upperbranch_nodes)?;
            d.set_item("gss_before_upper_interface_nodes", entry.gss_summary_before.interface_nodes)?;
            d.set_item("gss_before_lower_nodes", entry.gss_summary_before.lower_nodes)?;
            d.set_item("gss_before_lower_general_nodes", entry.gss_summary_before.lower_general_nodes)?;
            d.set_item("gss_before_lower_segment_nodes", entry.gss_summary_before.lower_segment_nodes)?;
            d.set_item("gss_before_total_unique_nodes", entry.gss_summary_before.total_unique_nodes)?;
            d.set_item("gss_before_total_edges", entry.gss_summary_before.total_edges)?;
            d.set_item("gss_before_accumulator_instances", entry.gss_summary_before.accumulator_instances)?;
            d.set_item("gss_before_max_depth", entry.gss_summary_before.max_depth)?;

            d.set_item("gss_path_count", gss_stacks_after_len)?;
            d.set_item("gss_top_values_count", entry.gss_summary_after.top_values_count)?;
            d.set_item("gss_upper_branch_nodes", entry.gss_summary_after.upperbranch_nodes)?;
            d.set_item("gss_upper_interface_nodes", entry.gss_summary_after.interface_nodes)?;
            d.set_item("gss_lower_nodes", entry.gss_summary_after.lower_nodes)?;
            d.set_item("gss_lower_general_nodes", entry.gss_summary_after.lower_general_nodes)?;
            d.set_item("gss_lower_segment_nodes", entry.gss_summary_after.lower_segment_nodes)?;
            d.set_item("gss_total_unique_nodes", entry.gss_summary_after.total_unique_nodes)?;
            d.set_item("gss_total_edges", entry.gss_summary_after.total_edges)?;
            d.set_item("gss_accumulator_instances", entry.gss_summary_after.accumulator_instances)?;
            d.set_item("gss_max_depth", entry.gss_summary_after.max_depth)?;
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
            d.set_item("det_ns", p.det_ns)?;
            d.set_item("nondet_ns", p.nondet_ns)?;
            d.set_item("nondet_det_ns", p.nondet_det_ns)?;
            d.set_item("det_exit_reason", p.det_exit_reason)?;
            d.set_item("det_exit_state", p.det_exit_state)?;
            d.set_item("n_det_action_lookups", p.n_det_action_lookups)?;
            d.set_item("n_det_goto_lookups", p.n_det_goto_lookups)?;
            d.set_item("n_det_popn_ops", p.n_det_popn_ops)?;
            d.set_item("n_nondet_reduce_ops", p.n_nondet_reduce_ops)?;
            d.set_item("n_nondet_merges", p.n_nondet_merges)?;
            d.set_item("n_nondet_isolates", p.n_nondet_isolates)?;
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
        commit_dict.set_item("exec_ns", commit_profile.exec_ns)?;
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
        commit_dict.set_item("adv_det_ns", commit_profile.adv_det_ns)?;
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
    m.add_function(wrap_pyfunction!(clear_weight_caches, m)?)?;
    m.add_function(wrap_pyfunction!(clear_stale_weights, m)?)?;
    m.add_function(wrap_pyfunction!(clear_weight_op_caches, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_grammar, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_grammar_glrm, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_terminals, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_grammar_def, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_prepared_grammar_def, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_glr_table, m)?)?;
    m.add_function(wrap_pyfunction!(compile_grammar_def_json, m)?)?;
    Ok(())
}

#[pyfunction]
fn clear_weight_caches() {
    glrmask::clear_weight_caches();
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
fn dump_json_schema_grammar(schema: &str) -> PyResult<String> {
    glrmask::dump_json_schema_grammar(schema)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn dump_json_schema_grammar_glrm(schema: &str) -> PyResult<String> {
    glrmask::dump_json_schema_grammar_glrm(schema)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn dump_json_schema_terminals(schema: &str) -> PyResult<String> {
    glrmask::dump_json_schema_terminals(schema)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn dump_json_schema_grammar_def(schema: &str) -> PyResult<String> {
    glrmask::dump_json_schema_grammar_def(schema)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn dump_json_schema_prepared_grammar_def(schema: &str) -> PyResult<String> {
    glrmask::dump_json_schema_prepared_grammar_def(schema)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

#[pyfunction]
fn dump_json_schema_glr_table(schema: &str) -> PyResult<String> {
    glrmask::dump_json_schema_glr_table(schema)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
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
