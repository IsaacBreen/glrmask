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
use pyo3::types::PyDict;
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
        for v in slice.iter_mut() {
            *v = 0;
        }
        let n = slice.len();
        let mut buf = vec![0u32; n];
        self.inner.with_dependent(|_owner, state| state.fill_mask(&mut buf));
        for (dst, src) in slice.iter_mut().zip(buf.iter()) {
            *dst = *src as i32;
        }
        Ok(())
    }

    fn commit_token(&mut self, token_id: u32) {
        self.inner
            .with_dependent_mut(|_owner, state| state.commit_token(token_id));
    }

    fn commit_tokens(&mut self, token_ids: Vec<u32>) {
        self.inner
            .with_dependent_mut(|_owner, state| state.commit_tokens(&token_ids));
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

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _glrmask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVocab>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyConstraintState>()?;
    Ok(())
}
