//! PyO3 Python bindings for glrmask.
//!
//! Exposes `Constraint` and `ConstraintState` to Python, matching the interface
//! expected by the CFA (constraint-framework-analysis) benchmarking harness.

use glrmask::ds::bitset::BitSet;
use numpy::{PyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// PyConstraint
// ---------------------------------------------------------------------------

/// Compiled grammar constraint. Immutable, thread-safe.
///
/// Create from a Lark grammar string + vocabulary:
///
/// ```python
/// vocab = {b"hello": 0, b"world": 1}
/// c = Constraint.from_lark("start: \"hello\" \"world\"", vocab, 1)
/// ```
#[pyclass(name = "Constraint")]
#[derive(Clone)]
pub struct PyConstraint {
    inner: Arc<glrmask::Constraint>,
    max_token: u32,
}

#[pymethods]
impl PyConstraint {
    /// Build from a Lark grammar string.
    ///
    /// Args:
    ///     lark_source: Lark-format grammar string.
    ///     token_to_id: dict mapping `bytes -> int` (token bytes to token ID).
    ///     max_token_id: maximum token ID in the vocabulary.
    #[staticmethod]
    fn from_lark(
        lark_source: &str,
        token_to_id: &Bound<'_, PyDict>,
        max_token_id: u32,
    ) -> PyResult<Self> {
        let vocab = dict_to_vocab(token_to_id, max_token_id)?;
        let constraint = glrmask::Constraint::from_lark(lark_source, &vocab)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: max_token_id,
        })
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
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: max_token_id,
        })
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
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: max_token_id,
        })
    }

    /// Save to bytes (bincode).
    fn save(&self) -> PyResult<Vec<u8>> {
        self.inner
            .save()
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Load from bytes (bincode).
    #[staticmethod]
    fn load(data: &[u8], max_token_id: u32) -> PyResult<Self> {
        let constraint = glrmask::Constraint::load(data)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(Self {
            inner: Arc::new(constraint),
            max_token: max_token_id,
        })
    }
}

// ---------------------------------------------------------------------------
// PyConstraintState
// ---------------------------------------------------------------------------

/// Mutable per-sequence state.
///
/// ```python
/// state = ConstraintState(constraint)
/// mask = state.get_mask_bv()
/// state.commit(token_id)
/// ```
#[pyclass(name = "ConstraintState")]
pub struct PyConstraintState {
    constraint: Arc<glrmask::Constraint>,
    state: glrmask::ConstraintState,
    max_token: u32,
}

#[pymethods]
impl PyConstraintState {
    #[new]
    fn new(constraint: &PyConstraint) -> Self {
        let state = constraint.inner.start();
        Self {
            constraint: constraint.inner.clone(),
            state,
            max_token: constraint.max_token,
        }
    }

    /// Commit a token ID to advance the state.
    fn commit(&mut self, token_id: u32) -> PyResult<()> {
        self.state
            .commit(&self.constraint, token_id)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Commit raw bytes (processes each byte through the tokenizer).
    fn commit_bytes(&mut self, data: &[u8]) -> PyResult<()> {
        // Process bytes one token at a time by finding the matching token
        // For now, commit each byte as if it were a single-byte token
        // This is a simplified version; a full implementation would do
        // tokenizer matching.
        for &_b in data {
            // Find token with these bytes
            // For now just error — the CFA adapter uses commit(token_id) path
            return Err(PyValueError::new_err(
                "commit_bytes not yet implemented; use commit(token_id) instead",
            ));
        }
        let _ = data;
        Ok(())
    }

    /// Get the token mask as a `PyBitset`.
    fn get_mask_bv(&self) -> PyBitset {
        let mask = self.state.compute_mask(&self.constraint);
        PyBitset { inner: mask }
    }

    /// Get the token mask as a boolean numpy array.
    fn get_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let mask = self.state.compute_mask(&self.constraint);
        let n = (self.max_token + 1) as usize;
        let mut bools = vec![false; n];
        for i in 0..n {
            if mask.get(i) {
                bools[i] = true;
            }
        }
        Ok(PyArray1::from_vec(py, bools))
    }

    /// Fill a pre-allocated int32 numpy array with the bitmask.
    ///
    /// Compatible with llguidance bitmask format: little-endian bit packing.
    fn fill_next_token_bitmask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
        let slice = bitmask.as_slice_mut().map_err(|e| {
            PyValueError::new_err(format!("Array must be contiguous: {e:?}"))
        })?;
        let mask = self.state.compute_mask(&self.constraint);
        // Zero the output first
        for v in slice.iter_mut() {
            *v = 0;
        }
        // Pack bits into i32 words (little-endian bit order)
        for i in 0..mask.len() {
            if mask.get(i) {
                let word_idx = i / 32;
                let bit_idx = i % 32;
                if word_idx < slice.len() {
                    slice[word_idx] |= 1i32 << bit_idx;
                }
            }
        }
        Ok(())
    }

    /// Returns the required buffer size in i32 elements for the mask.
    fn mask_buffer_size_i32(&self) -> usize {
        ((self.max_token as usize + 1) + 31) / 32
    }

    /// Whether the state is accepting (grammar fully matched).
    fn is_accepting(&self) -> bool {
        self.state.is_accepting(&self.constraint)
    }

    /// Whether the state is active (has any valid parse stacks).
    fn is_active(&self) -> bool {
        self.state.is_active()
    }
}

// ---------------------------------------------------------------------------
// PyBitset
// ---------------------------------------------------------------------------

/// Wraps a BitSet for Python.
#[pyclass(name = "Bitset")]
#[derive(Clone)]
pub struct PyBitset {
    inner: BitSet,
}

#[pymethods]
impl PyBitset {
    /// Return list of indices where the bit is set.
    fn to_indices(&self) -> Vec<usize> {
        self.inner.iter_ones().collect()
    }

    /// Return list of (start, end) ranges (inclusive on both ends).
    fn to_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut start: Option<usize> = None;
        let mut end: usize = 0;

        for i in self.inner.iter_ones() {
            match start {
                None => {
                    start = Some(i);
                    end = i;
                }
                Some(_) => {
                    if i == end + 1 {
                        end = i;
                    } else {
                        ranges.push((start.unwrap(), end));
                        start = Some(i);
                        end = i;
                    }
                }
            }
        }
        if let Some(s) = start {
            ranges.push((s, end));
        }
        ranges
    }

    fn __len__(&self) -> usize {
        self.inner.count_ones()
    }

    fn __contains__(&self, i: usize) -> bool {
        i < self.inner.len() && self.inner.get(i)
    }

    fn __repr__(&self) -> String {
        let ones: Vec<usize> = self.inner.iter_ones().take(10).collect();
        let total = self.inner.count_ones();
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

/// Convert a Python dict[bytes, int] to a glrmask::Vocab.
fn dict_to_vocab(
    token_to_id: &Bound<'_, PyDict>,
    _max_token_id: u32,
) -> PyResult<glrmask::Vocab> {
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
