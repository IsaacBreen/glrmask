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
    fn _debug_token_map(&self) -> Vec<u32> {
        self.inner.debug_original_token_to_internal()
    }

    /// Temporary diagnostic: return internal→originals mapping.
    fn _debug_internal_to_tokens(&self) -> Vec<Vec<u32>> {
        self.inner.debug_internal_token_to_tokens()
    }

    /// Walk bytes through DFA from every state. Returns list of (final_state, finalizers, futures).
    fn _debug_walk_dfa(&self, token_bytes: Vec<u8>) -> Vec<(u32, Vec<u32>, Vec<u32>)> {
        self.inner.debug_walk_dfa(&token_bytes)
    }

    /// Return action table entries for a given parser state.
    /// Returns list of (terminal_id, action_str) pairs.
    fn _debug_actions_for_state(&self, state: u32) -> Vec<(u32, String)> {
        self.inner.debug_actions_for_state(state)
    }

    /// Return rule info: list of (lhs_nonterminal, rhs_length, rhs_symbols_debug).
    fn _debug_rules(&self) -> Vec<(u32, usize, String)> {
        self.inner.debug_rules()
    }

    /// Return terminal names/mapping for the GLR table.
    fn _debug_num_terminals(&self) -> u32 {
        self.inner.debug_num_terminals()
    }

    fn _debug_num_states(&self) -> u32 {
        self.inner.debug_num_states()
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

    fn commit_token(&mut self, token_id: u32) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_token(token_id)))
    }

    fn commit_tokens(&mut self, token_ids: Vec<u32>) -> PyResult<()> {
        self.inner
            .with_dependent_mut(|_owner, state| string_result(state.commit_tokens(&token_ids)))
    }

    /// Like commit_token but returns profiling stats as a dict.
    fn commit_token_profiled<'py>(&mut self, py: Python<'py>, token_id: u32) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let (total_ns, scan_ns, prune_ns, queue_ns, fuse_ns, exec_ns, advance_ns, advance_may_check_ns, advance_core_ns, advance_future_disallow_ns, actionable_ns, may_advance_ns, n_tokenizer_states, n_queue_entries, n_advances,
             adv_n_reduces_above_floor, adv_n_floor_crossings, adv_n_nondet_waves, adv_n_nondet_branches, adv_clone_ns, adv_fast_path_ns, adv_det_ns, adv_nondet_ns, adv_vstack_len, adv_gss_depth,
             adv_det_exit_reason, adv_det_exit_state, adv_summary_ns,
             adv_n_det_action_lookups, adv_n_det_goto_lookups, adv_n_det_popn_ops,
             adv_n_nondet_reduce_ops, adv_n_nondet_merges, adv_n_nondet_isolates) =
            self.inner.with_dependent_mut(|_owner, state| {
                state.commit_token_profiled(token_id).map_err(|e| PyValueError::new_err(e))
            })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_ns", total_ns)?;
        dict.set_item("scan_ns", scan_ns)?;
        dict.set_item("prune_ns", prune_ns)?;
        dict.set_item("queue_ns", queue_ns)?;
        dict.set_item("fuse_ns", fuse_ns)?;
        dict.set_item("exec_ns", exec_ns)?;
        dict.set_item("advance_ns", advance_ns)?;
        dict.set_item("advance_may_check_ns", advance_may_check_ns)?;
        dict.set_item("advance_core_ns", advance_core_ns)?;
        dict.set_item("advance_future_disallow_ns", advance_future_disallow_ns)?;
        dict.set_item("actionable_ns", actionable_ns)?;
        dict.set_item("may_advance_ns", may_advance_ns)?;
        dict.set_item("n_tokenizer_states", n_tokenizer_states)?;
        dict.set_item("n_queue_entries", n_queue_entries)?;
        dict.set_item("n_advances", n_advances)?;
        dict.set_item("adv_n_reduces_above_floor", adv_n_reduces_above_floor)?;
        dict.set_item("adv_n_floor_crossings", adv_n_floor_crossings)?;
        dict.set_item("adv_n_nondet_waves", adv_n_nondet_waves)?;
        dict.set_item("adv_n_nondet_branches", adv_n_nondet_branches)?;
        dict.set_item("adv_clone_ns", adv_clone_ns)?;
        dict.set_item("adv_summary_ns", adv_summary_ns)?;
        dict.set_item("adv_fast_path_ns", adv_fast_path_ns)?;
        dict.set_item("adv_det_ns", adv_det_ns)?;
        dict.set_item("adv_nondet_ns", adv_nondet_ns)?;
        dict.set_item("adv_vstack_len", adv_vstack_len)?;
        dict.set_item("adv_gss_depth", adv_gss_depth)?;
        dict.set_item("adv_det_exit_reason", adv_det_exit_reason)?;
        dict.set_item("adv_det_exit_state", adv_det_exit_state)?;
        dict.set_item("adv_n_det_action_lookups", adv_n_det_action_lookups)?;
        dict.set_item("adv_n_det_goto_lookups", adv_n_det_goto_lookups)?;
        dict.set_item("adv_n_det_popn_ops", adv_n_det_popn_ops)?;
        dict.set_item("adv_n_nondet_reduce_ops", adv_n_nondet_reduce_ops)?;
        dict.set_item("adv_n_nondet_merges", adv_n_nondet_merges)?;
        dict.set_item("adv_n_nondet_isolates", adv_n_nondet_isolates)?;
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
