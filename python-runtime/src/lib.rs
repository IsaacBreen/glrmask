use numpy::{PyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn runtime_result<T>(result: glrmask_runtime::Result<T>) -> PyResult<T> {
    result.map_err(|error| PyValueError::new_err(error.to_string()))
}

fn state_result<T>(result: Result<T, String>) -> PyResult<T> {
    result.map_err(PyValueError::new_err)
}

#[pyclass(name = "Constraint")]
#[derive(Clone)]
pub struct PyRuntimeConstraint {
    artifact: glrmask_runtime::RuntimeArtifact,
    mask_len: usize,
}

#[pymethods]
impl PyRuntimeConstraint {
    #[staticmethod]
    fn load(data: &[u8]) -> PyResult<Self> {
        let artifact = runtime_result(glrmask_runtime::RuntimeArtifact::from_bytes(data.to_vec()))?;
        let session = runtime_result(glrmask_runtime::Session::from_artifact(artifact.clone()))?;
        Ok(Self { artifact, mask_len: session.mask_len() })
    }

    fn start(&self) -> PyResult<PyRuntimeConstraintState> {
        Ok(PyRuntimeConstraintState {
            inner: runtime_result(glrmask_runtime::Session::from_artifact(self.artifact.clone()))?,
        })
    }

    fn mask_len(&self) -> usize { self.mask_len }
}

#[pyclass(name = "ConstraintState")]
pub struct PyRuntimeConstraintState {
    inner: glrmask_runtime::Session,
}

#[pymethods]
impl PyRuntimeConstraintState {
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let words = self.inner.mask_words();
        let mut values = Vec::with_capacity(words.len() * 32);
        for word in words {
            let mut bits = word;
            for _ in 0..32 {
                values.push(bits & 1 != 0);
                bits >>= 1;
            }
        }
        Ok(PyArray1::from_vec(py, values))
    }

    fn mask_buffer_size_i32(&self) -> usize { self.inner.mask_len() }

    fn fill_mask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
        let slice = bitmask.as_slice_mut().map_err(|error| {
            PyValueError::new_err(format!("Array must be contiguous: {error:?}"))
        })?;
        let words = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        self.inner.fill_mask(words);
        Ok(())
    }

    fn fill_mask_timed_ns(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<u64> {
        let slice = bitmask.as_slice_mut().map_err(|error| {
            PyValueError::new_err(format!("Array must be contiguous: {error:?}"))
        })?;
        let words = unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u32, slice.len())
        };
        Ok(self.inner.fill_mask_timed_ns(words))
    }

    fn commit_token(&mut self, token_id: u32) -> PyResult<()> {
        state_result(self.inner.commit_token(token_id))
    }

    fn commit_token_timed_ns(&mut self, token_id: u32) -> PyResult<u64> {
        state_result(self.inner.commit_token_timed_ns(token_id))
    }

    fn is_finished(&self) -> bool { self.inner.is_finished() }

    fn eos_allowed(&self) -> bool { self.inner.eos_allowed() }

    fn reset(&mut self) { self.inner.reset(); }
}

#[pyo3::pymodule]
fn _glrmask_runtime(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRuntimeConstraint>()?;
    module.add_class::<PyRuntimeConstraintState>()?;
    Ok(())
}
