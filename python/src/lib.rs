//! PyO3 Python bindings for glrmask.
//!
//! Exposes `Constraint` and `ConstraintState` to Python, matching the interface
//! expected by the CFA (constraint-framework-analysis) benchmarking harness.

use numpy::{PyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::mem::ManuallyDrop;
use std::sync::Arc;

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
    /// Build from a Lark grammar string.
    #[staticmethod]
    fn from_lark(
        lark_source: &str,
        token_to_id: &Bound<'_, PyDict>,
        max_token_id: u32,
    ) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id, max_token_id)?;
        let constraint = glrmask::Constraint::from_lark(lark_source, &vocab)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner: Arc::new(constraint), max_token: max_token_id })
    }

    /// Build from an EBNF grammar string.
    #[staticmethod]
    fn from_ebnf(
        ebnf_source: &str,
        token_to_id: &Bound<'_, PyDict>,
        max_token_id: u32,
    ) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id, max_token_id)?;
        let constraint = glrmask::Constraint::from_ebnf(ebnf_source, &vocab)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner: Arc::new(constraint), max_token: max_token_id })
    }

    /// Build from a JSON Schema string.
    #[staticmethod]
    fn from_json_schema(
        schema: &str,
        token_to_id: &Bound<'_, PyDict>,
        max_token_id: u32,
    ) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id, max_token_id)?;
        let constraint = glrmask::Constraint::from_json_schema(schema, &vocab)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner: Arc::new(constraint), max_token: max_token_id })
    }

    /// Save to bytes (bincode). Infallible.
    fn save(&self) -> Vec<u8> {
        self.inner.save()
    }

    /// Load from bytes (bincode).
    #[staticmethod]
    fn load(data: &[u8], max_token_id: u32) -> PyResult<Self> {
        let constraint = glrmask::Constraint::load(data)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner: Arc::new(constraint), max_token: max_token_id })
    }
}

// ---------------------------------------------------------------------------
// OwnedState — encapsulated unsafe for PyConstraintState
//
// Problem: `ConstraintState<'a>` borrows `&'a Constraint`. PyO3 pyclass structs
// cannot carry lifetime parameters, so we cannot store `ConstraintState<'_>`
// directly in a pyclass field.
//
// Solution: borrow through the `Arc`'s stable heap pointer and manage drop order
// manually.  The invariants are:
//
//   I1. `*constraint` is heap-allocated and never moves (Arc guarantee).
//   I2. `state` holds a raw pointer to `*constraint`, disguised as the `'static`
//       lifetime by transmutation.
//   I3. `state` is dropped BEFORE `constraint`. We enforce this with
//       `ManuallyDrop<ConstraintState<'static>>` and an explicit `Drop` impl on
//       `OwnedState` that calls `drop(state)` before `drop(constraint)`.
//   I4. No borrow of `state` escapes `OwnedState` with a lifetime longer than
//       `&self`.  Every `ConstraintState` borrow in this module is scoped.
//
// Aliasing model: there is exactly one live borrow of `*constraint` at a time —
// through the `&'static Constraint` stored in `state.constraint`.  The `Arc`
// itself never mutates `*constraint` after the `Constraint` is constructed.
// `ConstraintState` only reads from `*constraint`; no mutation of the pointed-to
// `Constraint` ever occurs through the `state` path.
// ---------------------------------------------------------------------------

struct OwnedState {
    state: ManuallyDrop<glrmask::ConstraintState<'static>>,
    constraint: Arc<glrmask::Constraint>,
}

impl OwnedState {
    /// Create a new initial state pinned to `constraint`.
    fn new(constraint: Arc<glrmask::Constraint>) -> Self {
        // SAFETY: see struct-level comment.  Arc ensures `*constraint` lives at a
        // fixed address for at least as long as this `OwnedState` exists (because
        // we store the Arc here).  `state` is wrapped in ManuallyDrop and dropped
        // first in `OwnedState::drop`.
        let state = unsafe {
            let c: &'static glrmask::Constraint =
                &*(Arc::as_ptr(&constraint) as *const glrmask::Constraint);
            ManuallyDrop::new(c.start())
        };
        OwnedState { state, constraint }
    }

    /// Reset to the initial parse position.
    fn reset(&mut self) {
        // Drop the existing state first (so the old borrow is released),
        // then create a fresh one.
        unsafe { ManuallyDrop::drop(&mut self.state); }
        let fresh = unsafe {
            let c: &'static glrmask::Constraint =
                &*(Arc::as_ptr(&self.constraint) as *const glrmask::Constraint);
            ManuallyDrop::new(c.start())
        };
        self.state = fresh;
    }
}

impl Drop for OwnedState {
    fn drop(&mut self) {
        // I3: drop `state` (which holds the borrow) BEFORE `constraint` (the owner).
        unsafe { ManuallyDrop::drop(&mut self.state); }
        // `self.constraint` (the Arc) drops at the end of this function, after `state`.
    }
}

// ---------------------------------------------------------------------------
// PyConstraintState
// ---------------------------------------------------------------------------

/// Mutable per-sequence state.
#[pyclass(name = "ConstraintState")]
pub struct PyConstraintState {
    inner: OwnedState,
    max_token: u32,
}

#[pymethods]
impl PyConstraintState {
    #[new]
    fn new(constraint: &PyConstraint) -> Self {
        Self {
            inner: OwnedState::new(constraint.inner.clone()),
            max_token: constraint.max_token,
        }
    }

    /// Reset to the initial state without recompiling.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Commit a token ID (infallible — unknown token leads to empty next mask).
    fn commit(&mut self, token_id: u32) {
        self.inner.state.commit(token_id);
    }

    /// Commit raw bytes through the tokenizer DFA.
    fn commit_bytes(&mut self, data: &[u8]) {
        self.inner.state.commit_bytes(data);
    }

    /// Commit a list of token IDs.
    fn commit_tokens(&mut self, token_ids: Vec<u32>) {
        self.inner.state.commit_tokens(&token_ids);
    }

    /// Get the allowed-token mask as a PyBitset.
    fn get_mask_bv(&self) -> PyBitset {
        PyBitset { words: self.inner.state.mask(), total_tokens: self.max_token + 1 }
    }

    /// Get the allowed-token mask as a boolean numpy array.
    fn get_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let words = self.inner.state.mask();
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

    /// Fill a pre-allocated int32 numpy array with the packed bitmask.
    fn fill_next_token_bitmask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        for v in slice.iter_mut() {
            *v = 0;
        }
        let mut buf = vec![0u32; slice.len()];
        self.inner.state.fill_mask(&mut buf);
        for (dst, src) in slice.iter_mut().zip(buf.iter()) {
            *dst = *src as i32;
        }
        Ok(())
    }

    /// Number of i32 words needed for `fill_next_token_bitmask`.
    fn mask_buffer_size_i32(&self) -> usize {
        self.inner.constraint.mask_len()
    }

    /// Whether the grammar is fully satisfied (EOS valid).
    fn is_accepting(&self) -> bool {
        self.inner.state.is_finished()
    }

    /// Whether any continuation token is currently allowed.
    fn is_active(&self) -> bool {
        self.inner.state.mask().iter().any(|&w| w != 0) || self.inner.state.is_finished()
    }

    /// List of deterministically forced token IDs.
    fn get_forced_tokens(&self) -> Vec<u32> {
        self.inner.state.force()
    }
}

// ---------------------------------------------------------------------------
// PyBitset
// ---------------------------------------------------------------------------

/// Packed bitmask returned by `get_mask_bv()`.
#[pyclass(name = "Bitset")]
#[derive(Clone)]
pub struct PyBitset {
    words: Vec<u32>,
    total_tokens: u32,
}

#[pymethods]
impl PyBitset {
    /// Return sorted list of allowed token IDs.
    fn to_indices(&self) -> Vec<usize> {
        let limit = self.total_tokens as usize;
        let mut out = Vec::new();
        for (wi, &word) in self.words.iter().enumerate() {
            if word == 0 { continue; }
            for bit in 0..32u32 {
                if (word >> bit) & 1 != 0 {
                    let id = wi * 32 + bit as usize;
                    if id < limit { out.push(id); }
                }
            }
        }
        out
    }

    /// Return list of (start, end) inclusive ranges of allowed token IDs.
    fn to_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut run: Option<(usize, usize)> = None;
        for id in self.to_indices() {
            match run {
                None => { run = Some((id, id)); }
                Some((s, e)) if id == e + 1 => { run = Some((s, id)); }
                Some((s, e)) => { ranges.push((s, e)); run = Some((id, id)); }
            }
        }
        if let Some((s, e)) = run { ranges.push((s, e)); }
        ranges
    }

    fn __len__(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn __contains__(&self, i: usize) -> bool {
        let (wi, bi) = (i / 32, i % 32);
        wi < self.words.len() && (self.words[wi] >> bi) & 1 != 0
    }

    fn __repr__(&self) -> String {
        let ones: Vec<usize> = self.to_indices().into_iter().take(10).collect();
        let total = self.__len__();
        if total <= 10 { format!("Bitset({ones:?})") }
        else { format!("Bitset({ones:?}... [{total} set])") }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dict_to_vocab(token_to_id: &Bound<'_, PyDict>, _max_token_id: u32) -> PyResult<glrmask::Vocab> {
    let mut entries = Vec::new();
    for (key, value) in token_to_id.iter() {
        let token_bytes: Vec<u8> = key.extract()?;
        let token_id: u32 = value.extract()?;
        entries.push((token_id, token_bytes));
    }
    Ok(glrmask::Vocab::new(entries, None))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _glrmask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyConstraintState>()?;
    m.add_class::<PyBitset>()?;
    Ok(())
}
