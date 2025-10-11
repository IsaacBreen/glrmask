use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyType};

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::core::{LeveledGSS as CoreLeveledGSS, LeveledGSSStats as CoreLeveledGSSStats, Merge};

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

#[pyclass(name = "LeveledGSSStats", module = "leveled_gss_rs", unsendable)]
struct LeveledGSSStats {
    #[pyo3(get)]
    top_values: PyObject,
    #[pyo3(get)]
    num_upperbranch_nodes: usize,
    #[pyo3(get)]
    num_interface_nodes: usize,
    #[pyo3(get)]
    num_lower_nodes: usize,
    #[pyo3(get)]
    total_unique_nodes: usize,
    #[pyo3(get)]
    upper_edges: usize,
    #[pyo3(get)]
    interface_to_lower_edges: usize,
    #[pyo3(get)]
    lower_edges: usize,
    #[pyo3(get)]
    total_edges: usize,
    #[pyo3(get)]
    max_upper_depth: isize,
    #[pyo3(get)]
    max_lower_depth: isize,
    #[pyo3(get)]
    distinct_values_count: usize,
    #[pyo3(get)]
    distinct_values: PyObject,
    #[pyo3(get)]
    unique_accumulators_count: usize,
    #[pyo3(get)]
    unique_accumulators: PyObject,
    #[pyo3(get)]
    total_accumulator_instances: usize,
    #[pyo3(get)]
    num_upper_with_empty: usize,
    #[pyo3(get)]
    num_interfaces_with_empty: usize,
    #[pyo3(get)]
    num_lower_terminal_nodes: usize,
    #[pyo3(get)]
    num_interface_implicit_terminals: usize,
    #[pyo3(get)]
    num_multi_depth_slots_upper: usize,
    #[pyo3(get)]
    num_multi_depth_slots_lower: usize,
    #[pyo3(get)]
    max_multiplicity_per_value_upper: usize,
    #[pyo3(get)]
    max_multiplicity_per_value_lower: usize,
    #[pyo3(get)]
    average_in_degree: f64,
    #[pyo3(get)]
    max_in_degree: usize,
    #[pyo3(get)]
    structural_sharing_factor: f64,
    #[pyo3(get)]
    promotable_upper_nodes: usize,
}

#[pymethods]
impl LeveledGSSStats {
    fn __str__(&self, py: Python) -> PyResult<String> {
        fn fmt_subset(py: Python, set: &PySet, max_items: usize) -> PyResult<String> {
            if set.is_empty() {
                return Ok("{}".to_string());
            }
            let mut items: Vec<String> = Vec::new();
            for (i, item) in set.iter().enumerate() {
                if i >= max_items {
                    break;
                }
                items.push(item.repr()?.to_str()?.to_string());
            }
            let suffix = if set.len() > max_items { ", ..." } else { "" };
            Ok(format!("{{{}}}{}", items.join(", "), suffix))
        }

        let mut lines: Vec<String> = Vec::new();
        lines.push("LeveledGSSStats".to_string());
        lines.push(format!(
            "- top_values: {} distinct -> {}",
            self.top_values.as_ref(py).downcast::<PySet>()?.len(),
            fmt_subset(py, self.top_values.as_ref(py).downcast()?, 10)?
        ));

        lines.push("- structure:".to_string());
        lines.push(format!("  nodes: UpperBranch={}, Interface={}, Lower={}, total={}", self.num_upperbranch_nodes, self.num_interface_nodes, self.num_lower_nodes, self.total_unique_nodes));
        lines.push(format!("  edges: upper={}, interface_to_lower={}, lower={}, total={}", self.upper_edges, self.interface_to_lower_edges, self.lower_edges, self.total_edges));
        lines.push(format!("  depths: max_upper_depth={}, max_lower_depth={}", self.max_upper_depth, self.max_lower_depth));

        let distinct_values_set: &PySet = self.distinct_values.as_ref(py).downcast()?;

        lines.push("- values/accumulators:".to_string());
        lines.push(format!("  distinct_values_count={}, sample={}", self.distinct_values_count, fmt_subset(py, distinct_values_set, 10)?));
        lines.push(format!("  unique_accumulators_count={} (physically stored)", self.unique_accumulators_count));
        lines.push(format!("  total_accumulator_instances={} (storage slots used)", self.total_accumulator_instances));

        lines.push("- empties/terminals:".to_string());
        lines.push(format!("  upper_with_empty={} (nodes representing a true empty stack)", self.num_upper_with_empty));
        lines.push(format!("  interfaces_with_empty={} (nodes representing a true empty stack)", self.num_interfaces_with_empty));
        lines.push(format!("  lower_terminal_nodes={} (nodes where a stack can end)", self.num_lower_terminal_nodes));
        lines.push(format!("  interface_implicit_terminals={} (interfaces with no children)", self.num_interface_implicit_terminals));

        lines.push("- multi-depth slots:".to_string());
        lines.push(format!("  num_multi_depth_slots_upper={}, max_multiplicity_per_value_upper={}", self.num_multi_depth_slots_upper, self.max_multiplicity_per_value_upper));
        lines.push(format!("  num_multi_depth_slots_lower={}, max_multiplicity_per_value_lower={}", self.num_multi_depth_slots_lower, self.max_multiplicity_per_value_lower));

        lines.push("- sharing/graph:".to_string());
        lines.push(format!("  average_in_degree={}, max_in_degree={}, structural_sharing_factor={}", self.average_in_degree, self.max_in_degree, self.structural_sharing_factor));

        lines.push("- canonicalization opportunities (non-fatal):".to_string());
        lines.push(format!("  promotable_upper_nodes={}", self.promotable_upper_nodes));

        Ok(lines.join("\n"))
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

    #[pyo3(signature = (min_len = None, max_len = None))]
    fn filter_by_length(&self, min_len: Option<isize>, max_len: Option<isize>) -> PyResult<Self> {
        Ok(LeveledGSS {
            inner: self.inner.filter_by_length(min_len, max_len),
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

    #[pyo3(signature = (levels = None))]
    fn fuse(&self, levels: Option<isize>) -> PyResult<Self> {
        Ok(LeveledGSS {
            inner: self.inner.fuse(levels),
        })
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

    fn stats(&self) -> PyResult<LeveledGSSStats> {
        Python::with_gil(|py| {
            let rust_stats = self.inner.stats();

            let top_values = PySet::new(
                py,
                &rust_stats
                    .top_values
                    .into_iter()
                    .map(|k| k.0)
                    .collect::<Vec<_>>(),
            )?
            .to_object(py);
            let distinct_values = PySet::new(
                py,
                &rust_stats
                    .distinct_values
                    .into_iter()
                    .map(|k| k.0)
                    .collect::<Vec<_>>(),
            )?
            .to_object(py);
            let unique_accumulators = PySet::new(
                py,
                &rust_stats
                    .unique_accumulators
                    .into_iter()
                    .map(|a| a.0)
                    .collect::<Vec<_>>(),
            )?
            .to_object(py);

            Ok(LeveledGSSStats {
                top_values,
                num_upperbranch_nodes: rust_stats.num_upperbranch_nodes,
                num_interface_nodes: rust_stats.num_interface_nodes,
                num_lower_nodes: rust_stats.num_lower_nodes,
                total_unique_nodes: rust_stats.total_unique_nodes,
                upper_edges: rust_stats.upper_edges,
                interface_to_lower_edges: rust_stats.interface_to_lower_edges,
                lower_edges: rust_stats.lower_edges,
                total_edges: rust_stats.total_edges,
                max_upper_depth: rust_stats.max_upper_depth,
                max_lower_depth: rust_stats.max_lower_depth,
                distinct_values_count: rust_stats.distinct_values_count,
                distinct_values,
                unique_accumulators_count: rust_stats.unique_accumulators_count,
                unique_accumulators,
                total_accumulator_instances: rust_stats.total_accumulator_instances,
                num_upper_with_empty: rust_stats.num_upper_with_empty,
                num_interfaces_with_empty: rust_stats.num_interfaces_with_empty,
                num_lower_terminal_nodes: rust_stats.num_lower_terminal_nodes,
                num_interface_implicit_terminals: rust_stats.num_interface_implicit_terminals,
                num_multi_depth_slots_upper: rust_stats.num_multi_depth_slots_upper,
                num_multi_depth_slots_lower: rust_stats.num_multi_depth_slots_lower,
                max_multiplicity_per_value_upper: rust_stats.max_multiplicity_per_value_upper,
                max_multiplicity_per_value_lower: rust_stats.max_multiplicity_per_value_lower,
                average_in_degree: rust_stats.average_in_degree,
                max_in_degree: rust_stats.max_in_degree,
                structural_sharing_factor: rust_stats.structural_sharing_factor,
                promotable_upper_nodes: rust_stats.promotable_upper_nodes,
            })
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
    m.add_class::<LeveledGSSStats>()?;
    Ok(())
}

