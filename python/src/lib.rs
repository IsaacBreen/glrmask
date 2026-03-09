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
//
// Owner: `Arc<glrmask::Constraint>` — heap-stable, cheap to clone (for reset).
// Dependent: `glrmask::ConstraintState<'owner>` — borrows the dereffed Constraint.
//
// `#[not_covariant]` is required here because we need mutable access via
// `with_dependent_mut`. (Declaring a covariant type as not_covariant is safe —
// it just forgoes the covariance optimisation.)
//
// `self_cell!` requires the dependent to be a plain identifier, so we introduce
// a type alias below.
// ---------------------------------------------------------------------------

// Type alias required by `self_cell!` macro (it expects `$Dependent:ident`).
type ConstraintState<'a> = glrmask::ConstraintState<'a>;

self_cell!(
    struct OwnedState {
        owner: Arc<glrmask::Constraint>,
        #[not_covariant]
        dependent: ConstraintState,
    }
);

impl OwnedState {
    /// Build a fresh initial state from the given `Arc`.
    fn from_arc(arc: Arc<glrmask::Constraint>) -> Self {
        // `arc_ref` is `&Arc<Constraint>`; deref into `&Constraint` via `Arc::Deref`.
        // The resulting `ConstraintState<'_>` borrows through `arc_ref` whose
        // lifetime is tied to the cell — `self_cell` ensures `owner` outlives
        // `dependent`.
        OwnedState::new(arc, |arc_ref| arc_ref.start())
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
    /// Build from a Lark grammar string.
    #[staticmethod]
    fn from_lark(
        lark_source: &str,
        token_to_id: &Bound<'_, PyDict>,
        max_token_id: u32,
    ) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id)?;
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
        let vocab = dict_to_vocab(token_to_id)?;
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
        let vocab = dict_to_vocab(token_to_id)?;
        let constraint = glrmask::Constraint::from_json_schema(schema, &vocab)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner: Arc::new(constraint), max_token: max_token_id })
    }

    /// Serialise to bytes (bincode). Infallible.
    fn save(&self) -> Vec<u8> {
        self.inner.save()
    }

    /// Deserialise from bytes (bincode).
    #[staticmethod]
    fn load(data: &[u8], max_token_id: u32) -> PyResult<Self> {
        let constraint = glrmask::Constraint::load(data)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self { inner: Arc::new(constraint), max_token: max_token_id })
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
    #[new]
    fn new(constraint: &PyConstraint) -> Self {
        Self {
            inner: OwnedState::from_arc(constraint.inner.clone()),
            max_token: constraint.max_token,
        }
    }

    /// Reset to the initial parse position.
    fn reset(&mut self) {
        // Clone the Arc (O(1)), then replace the cell with a fresh initial state.
        let arc = self.inner.borrow_owner().clone();
        self.inner = OwnedState::from_arc(arc);
    }

    /// Commit a token ID (infallible — unknown token leads to an empty next mask).
    fn commit(&mut self, token_id: u32) {
        self.inner
            .with_dependent_mut(|_owner, state| state.commit_token(token_id));
    }

    /// Commit raw bytes through the tokenizer DFA.
    fn commit_bytes(&mut self, data: &[u8]) {
        self.inner.with_dependent_mut(|_owner, state| state.commit_bytes(data));
    }

    /// Commit a list of token IDs.
    fn commit_tokens(&mut self, token_ids: Vec<u32>) {
        self.inner.with_dependent_mut(|_owner, state| state.commit_tokens(&token_ids));
    }

    /// Get the allowed-token mask as a `Bitset` object.
    fn get_mask_bv(&self) -> PyBitset {
        let words = self.inner.with_dependent(|_owner, state| state.mask());
        PyBitset { words, total_tokens: self.max_token + 1 }
    }

    /// Get the allowed-token mask as a boolean numpy array.
    fn get_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
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

    /// Fill a pre-allocated int32 numpy array with the packed bitmask.
    fn fill_next_token_bitmask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
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

    /// Number of i32 words needed for `fill_next_token_bitmask`.
    fn mask_buffer_size_i32(&self) -> usize {
        self.inner.borrow_owner().mask_len()
    }

    /// Whether the grammar is fully satisfied (EOS is valid next token).
    fn is_accepting(&self) -> bool {
        self.inner.with_dependent(|_owner, state| state.is_finished())
    }

    /// Whether any continuation token is currently allowed (or EOS is valid).
    fn is_active(&self) -> bool {
        self.inner.with_dependent(|_owner, state| {
            state.mask().iter().any(|&w| w != 0) || state.is_finished()
        })
    }

    /// List of deterministically forced token IDs (or empty if ambiguous/blocked).
    fn get_forced_tokens(&self) -> Vec<u32> {
        self.inner.with_dependent(|_owner, state| state.force())
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
    /// Return a sorted list of allowed token IDs.
    fn to_indices(&self) -> Vec<usize> {
        let limit = self.total_tokens as usize;
        let mut out = Vec::new();
        for (wi, &word) in self.words.iter().enumerate() {
            if word == 0 {
                continue;
            }
            for bit in 0..32u32 {
                if (word >> bit) & 1 != 0 {
                    let id = wi * 32 + bit as usize;
                    if id < limit {
                        out.push(id);
                    }
                }
            }
        }
        out
    }

    /// Return list of `(start, end)` inclusive ranges of allowed token IDs.
    fn to_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut run: Option<(usize, usize)> = None;
        for id in self.to_indices() {
            match run {
                None => {
                    run = Some((id, id));
                }
                Some((s, e)) if id == e + 1 => {
                    run = Some((s, id));
                }
                Some((s, e)) => {
                    ranges.push((s, e));
                    run = Some((id, id));
                }
            }
        }
        if let Some((s, e)) = run {
            ranges.push((s, e));
        }
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
        if total <= 10 {
            format!("Bitset({ones:?})")
        } else {
            format!("Bitset({ones:?}... [{total} set])")
        }
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
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _glrmask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyConstraintState>()?;
    m.add_class::<PyBitset>()?;
    Ok(())
}
