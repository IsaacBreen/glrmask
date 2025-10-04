use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyType};

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::core::{LeveledGSS as CoreLeveledGSS, Merge};

mod core;

// Wrapper types for Python interop used only in this binding layer.

#[derive(Clone)]
struct PyKey(PyObject);

impl PartialEq for PyKey {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            match self.0.as_ref(py).rich_compare(other.0.as_ref(py), CompareOp::Eq) {
                Ok(res) => res.is_true().unwrap_or(false),
                Err(_) => self.0.as_ref(py).is(other.0.as_ref(py)),
            }
        })
    }
}
impl Eq for PyKey {}

impl Hash for PyKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Python::with_gil(|py| match self.0.as_ref(py).hash() {
            Ok(h) => h.hash(state),
            Err(_) => 0.hash(state),
        });
    }
}

#[derive(Clone)]
struct PyAcc(PyObject);

impl PartialEq for PyAcc {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            match self.0.as_ref(py).rich_compare(other.0.as_ref(py), CompareOp::Eq) {
                Ok(res) => res.is_true().unwrap_or(false),
                Err(_) => self.0.as_ref(py).is(other.0.as_ref(py)),
            }
        })
    }
}
impl Eq for PyAcc {}

impl Hash for PyAcc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Python::with_gil(|py| match self.0.as_ref(py).hash() {
            Ok(h) => h.hash(state),
            Err(_) => 0.hash(state),
        });
    }
}

impl Merge for PyAcc {
    fn merge(&self, other: &Self) -> Self {
        Python::with_gil(|py| {
            let merged = self
                .0
                .call_method1(py, "merge", (other.0.clone_ref(py),))
                .expect("Accumulator merge failed in Python");
            PyAcc(merged)
        })
    }
}

#[pyclass(name = "LeveledGSS", module = "leveled_gss_rs", unsendable)]
#[derive(Clone)]
struct LeveledGSS {
    inner: CoreLeveledGSS<PyKey, PyAcc>,
}

#[pymethods]
impl LeveledGSS {
    #[new]
    fn new() -> Self {
        LeveledGSS {
            inner: CoreLeveledGSS::empty(),
        }
    }

    #[classmethod]
    fn empty(_cls: &PyType) -> Self {
        LeveledGSS {
            inner: CoreLeveledGSS::empty(),
        }
    }

    #[classmethod]
    fn from_stacks(_cls: &PyType, stacks: &PyList) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Canonicalize via ReferenceGSS to match Python semantics exactly
            let reference_gss_module = py.import("gss_tester.implementations.reference_impl")?;
            let ref_gss_class = reference_gss_module.getattr("ReferenceGSS")?;
            let gss_instance = ref_gss_class.call1((stacks,))?;
            let canonical_stacks: &PyList = gss_instance.getattr("_stacks")?.downcast()?;

            let mut rust_stacks: Vec<(Vec<PyKey>, PyAcc)> = Vec::with_capacity(canonical_stacks.len());
            for item in canonical_stacks {
                let tuple: &PyTuple = item.downcast()?;
                let vals: &PyList = tuple.get_item(0)?.downcast()?;
                let acc: PyObject = tuple.get_item(1)?.to_object(py);

                let vec_vals: Vec<PyKey> = vals.iter().map(|v| PyKey(v.to_object(py))).collect();
                rust_stacks.push((vec_vals, PyAcc(acc)));
            }

            Ok(LeveledGSS {
                inner: CoreLeveledGSS::from_stacks(&rust_stacks),
            })
        })
    }

    fn to_stacks(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stacks = self.inner.to_stacks();
            let res = PyList::empty(py);
            for (vals, acc) in stacks {
                let py_vals = PyList::new(py, vals.into_iter().map(|k| k.0));
                res.append(PyTuple::new(py, &[py_vals.to_object(py), acc.0]))?;
            }

            // Canonicalize and sort via ReferenceGSS
            let reference_gss_module = py.import("gss_tester.implementations.reference_impl")?;
            let ref_gss_class = reference_gss_module.getattr("ReferenceGSS")?;
            let gss_instance = ref_gss_class.call1((res.to_object(py),))?;
            gss_instance
                .call_method0("to_stacks")
                .map(|any| any.to_object(py))
        })
    }

    fn push(&self, value: PyObject) -> PyResult<Self> {
        Ok(LeveledGSS {
            inner: self.inner.push(PyKey(value)),
        })
    }

    #[classmethod]
    fn push_many(_cls: &PyType, items: &PyList) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut dest = LeveledGSS {
                inner: CoreLeveledGSS::empty(),
            };
            for item in items.iter() {
                let tuple: &PyTuple = item.downcast()?;
                let gss_item: PyRef<LeveledGSS> = tuple.get_item(0)?.extract()?;
                let value: PyObject = tuple.get_item(1)?.to_object(py);
                let pushed = gss_item.push(value)?;
                dest = dest.merge(&pushed)?;
            }
            Ok(dest)
        })
    }

    fn pop(&self) -> PyResult<Self> {
        Ok(LeveledGSS {
            inner: self.inner.pop(),
        })
    }

    fn popn(&self, n: isize) -> PyResult<Self> {
        Ok(LeveledGSS {
            inner: self.inner.popn(n),
        })
    }

    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.inner.is_empty())
    }

    fn isolate(&self, value: Option<PyObject>) -> PyResult<Self> {
        Ok(LeveledGSS {
            inner: self.inner.isolate(value.map(PyKey)),
        })
    }

    fn isolate_many(&self, values: &PySet) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut set: HashSet<Option<PyKey>> = HashSet::new();
            for v in values.iter() {
                if v.is_none() {
                    set.insert(None);
                } else {
                    set.insert(Some(PyKey(v.to_object(py))));
                }
            }
            Ok(LeveledGSS {
                inner: self.inner.isolate_many(set.into_iter()),
            })
        })
    }

    #[pyo3(signature = (func, memo = None))]
    fn apply(&self, func: PyObject, memo: Option<&PyDict>) -> PyResult<Self> {
        // We ignore the external memo and rely on internal per-accumulator memoization.
        let func_object = func.clone();
        let new_inner = self.inner.apply::<PyAcc, _>(move |acc: &PyAcc| {
            Python::with_gil(|py| {
                let out = func_object
                    .call1(py, (acc.0.clone_ref(py),))
                    .expect("apply() callback raised");
                PyAcc(out)
            })
        });
        Ok(LeveledGSS { inner: new_inner })
    }

    #[pyo3(signature = (predicate, memo = None))]
    fn prune(&self, predicate: PyObject, memo: Option<&PyDict>) -> PyResult<Self> {
        let pred_object = predicate.clone();
        let new_inner = self.inner.prune(move |acc: &PyAcc| {
            Python::with_gil(|py| {
                pred_object
                    .call1(py, (acc.0.clone_ref(py),))
                    .expect("prune() callback raised")
                    .is_true(py)
                    .expect("predicate.__bool__ failed")
            })
        });
        Ok(LeveledGSS { inner: new_inner })
    }

    #[pyo3(signature = (mutator, memo = None))]
    fn apply_and_prune(&self, mutator: PyObject, memo: Option<&PyDict>) -> PyResult<Self> {
        let mutator_obj = mutator.clone();
        let new_inner = self.inner.apply_and_prune::<PyAcc, _>(move |acc: &PyAcc| {
            Python::with_gil(|py| {
                let r = mutator_obj
                    .call1(py, (acc.0.clone_ref(py),))
                    .expect("apply_and_prune() callback raised");
                if r.is_none(py) {
                    None
                } else {
                    Some(PyAcc(r))
                }
            })
        });
        Ok(LeveledGSS { inner: new_inner })
    }

    fn merge(&self, other: &Self) -> PyResult<Self> {
        Ok(LeveledGSS {
            inner: self.inner.merge(&other.inner),
        })
    }

    #[classmethod]
    fn merge_many(_cls: &PyType, gss_list: &PyList) -> PyResult<Self> {
        let mut accumulator = LeveledGSS {
            inner: CoreLeveledGSS::empty(),
        };
        for item in gss_list.iter() {
            let gss: PyRef<LeveledGSS> = item.extract()?;
            accumulator = accumulator.merge(&gss)?;
        }
        Ok(accumulator)
    }

    fn peek(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let keys: Vec<PyObject> = self
                .inner
                .peek()
                .into_iter()
                .map(|k| k.0)
                .collect();
            Ok(PySet::new(py, &keys)?.to_object(py))
        })
    }

    fn reduce_acc(&self) -> PyResult<Option<PyObject>> {
        Ok(self.inner.reduce_acc().map(|a| a.0))
    }

    fn to_reference_impl(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stacks = self.to_stacks()?;
            let reference_gss_module = py.import("gss_tester.implementations.reference_impl")?;
            let ref_gss_class = reference_gss_module.getattr("ReferenceGSS")?;
            ref_gss_class
                .call_method1("from_stacks", (stacks, ))
                .map(|any| any.to_object(py))
        })
    }

    fn __str__(&self) -> PyResult<String> {
        let stacks = self.to_stacks()?;
        Ok(format!("LeveledGSS({})", stacks))
    }

    fn __repr__(&self) -> PyResult<String> {
        let stacks = self.to_stacks()?;
        Python::with_gil(|py| {
            let repr = stacks.as_ref(py).repr()?.to_str()?;
            Ok(format!("LeveledGSS({})", repr))
        })
    }
}

#[pymodule]
fn leveled_gss_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LeveledGSS>()?;
    Ok(())
}
