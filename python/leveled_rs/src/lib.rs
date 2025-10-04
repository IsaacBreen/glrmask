use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyType};
use pyo3::exceptions::PyValueError;
use pyo3::basic::CompareOp;
use std::sync::Arc;
use im::{HashMap, OrdMap};
use std::hash::{Hash, Hasher};
use std::collections::VecDeque;

// Wrapper for PyObject to be used as a key in HashMap
#[derive(Clone)]
struct PyObjectWrapper(PyObject);

impl PartialEq for PyObjectWrapper {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            match self.0.as_ref(py).rich_compare(other.0.as_ref(py), CompareOp::Eq) {
                Ok(res) => res.is_true().unwrap_or(false),
                Err(_) => self.0.as_ref(py).is(other.0.as_ref(py)), // Fallback to pointer comparison
            }
        })
    }
}

impl Eq for PyObjectWrapper {}

impl Hash for PyObjectWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Python::with_gil(|py| {
            match self.0.as_ref(py).hash() {
                Ok(hash_val) => hash_val.hash(state),
                Err(_) => 0.hash(state), // Fallback for unhashable types
            }
        });
    }
}

type Children<N> = HashMap<PyObjectWrapper, OrdMap<isize, Arc<N>>>;

#[derive(Clone)]
struct Lower {
    children: Children<Lower>,
    empty: bool,
    max_depth: isize,
}

impl PartialEq for Lower {
    fn eq(&self, other: &Self) -> bool {
        self.empty == other.empty && self.max_depth == other.max_depth && self.children.ptr_eq(&other.children)
    }
}
impl Eq for Lower {}

impl Hash for Lower {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.empty.hash(state);
        // self.max_depth.hash(state); // max_depth is derived, shouldn't be part of hash
        self.children.iter().for_each(|(k, v)| {
            k.hash(state);
            v.iter().for_each(|(d, n)| {
                d.hash(state);
                Arc::as_ptr(n).hash(state);
            });
        });
    }
}

fn py_obj_eq(a: &Option<PyObject>, b: &Option<PyObject>) -> bool {
    Python::with_gil(|py| match (a, b) {
        (Some(a_obj), Some(b_obj)) => a_obj.as_ref(py).rich_compare(b_obj.as_ref(py), CompareOp::Eq).map_or(false, |c| c.is_true().unwrap_or(false)),
        (None, None) => true,
        _ => false,
    })
}

#[derive(Clone)]
struct Interface {
    children: Children<Lower>,
    acc: PyObject,
    empty: Option<PyObject>,
    max_depth: isize,
}

impl PartialEq for Interface {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            self.acc.as_ref(py).eq(other.acc.as_ref(py)).unwrap_or(false) &&
            py_obj_eq(&self.empty, &other.empty) &&
            self.max_depth == other.max_depth &&
            self.children.ptr_eq(&other.children)
        })
    }
}
impl Eq for Interface {}

impl Hash for Interface {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Python::with_gil(|py| {
            self.acc.as_ref(py).hash().unwrap_or(0).hash(state);
            self.empty.as_ref().and_then(|e| e.as_ref(py).hash().ok()).hash(state);
        });
        // self.max_depth.hash(state);
        self.children.iter().for_each(|(k, v)| {
            k.hash(state);
            v.iter().for_each(|(d, n)| {
                d.hash(state);
                Arc::as_ptr(n).hash(state);
            });
        });
    }
}

#[derive(Clone)]
struct UpperBranch {
    children: Children<Upper>,
    empty: Option<PyObject>,
    max_depth: isize,
}

impl PartialEq for UpperBranch {
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| {
            py_obj_eq(&self.empty, &other.empty) &&
            self.max_depth == other.max_depth &&
            self.children.ptr_eq(&other.children)
        })
    }
}
impl Eq for UpperBranch {}

impl Hash for UpperBranch {
     fn hash<H: Hasher>(&self, state: &mut H) {
        Python::with_gil(|py| {
            self.empty.as_ref().and_then(|e| e.as_ref(py).hash().ok()).hash(state);
        });
        // self.max_depth.hash(state);
        self.children.iter().for_each(|(k, v)| {
            k.hash(state);
            v.iter().for_each(|(d, n)| {
                d.hash(state);
                Arc::as_ptr(n).hash(state);
            });
        });
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Upper {
    Branch(Arc<UpperBranch>),
    Interface(Arc<Interface>),
}

impl Upper {
    fn max_depth(&self) -> isize {
        match self {
            Upper::Branch(b) => b.max_depth,
            Upper::Interface(i) => i.max_depth,
        }
    }

    fn children(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let keys: Vec<PyObject> = match self {
                Upper::Branch(b) => b.children.keys().map(|k| k.0.clone_ref(py)).collect(),
                Upper::Interface(i) => i.children.keys().map(|k| k.0.clone_ref(py)).collect(),
            };
            Ok(keys)
        })
    }
}

#[pyclass(name = "LeveledGSS", module="leveled_gss_rs", unsendable)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct LeveledGSS {
    inner: Arc<Upper>,
}

fn get_max_depth_lower(children: &Children<Lower>) -> isize {
    children.values().flat_map(|kids| kids.values()).map(|c| c.max_depth).max().map_or(0, |d| d + 1)
}

fn get_max_depth_upper(children: &Children<Upper>) -> isize {
    children.values().flat_map(|kids| kids.values()).map(|c| c.max_depth()).max().map_or(0, |d| d + 1)
}

impl LeveledGSS {
    fn _empty() -> Self {
        let inner = Arc::new(Upper::Branch(Arc::new(UpperBranch {
            children: HashMap::new(),
            empty: None,
            max_depth: 0,
        })));
        LeveledGSS { inner }
    }
}

#[pymethods]
impl LeveledGSS {
    #[new]
    fn new() -> Self {
        LeveledGSS::_empty()
    }

    #[classmethod]
    fn empty(_cls: &PyType) -> Self {
        LeveledGSS::_empty()
    }

    fn to_stacks(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let res = PyList::empty(py);

            fn dfs_lower(l: &Lower, pref: &mut Vec<PyObject>, acc: &PyObject, res: &PyList, py: Python) -> PyResult<()> {
                if l.empty {
                    let stack = PyList::new(py, pref.iter().rev());
                    res.append(PyTuple::new(py, &[stack.to_object(py), acc.clone_ref(py)]))?;
                }
                for (v, kids) in l.children.iter() {
                    for child in kids.values() {
                        pref.push(v.0.clone_ref(py));
                        dfs_lower(child, pref, acc, res, py)?;
                        pref.pop();
                    }
                }
                Ok(())
            }

            fn dfs_upper(u: &Upper, pref: &mut Vec<PyObject>, res: &PyList, py: Python) -> PyResult<()> {
                match u {
                    Upper::Branch(b) => {
                        if let Some(empty_acc) = &b.empty {
                            let stack = PyList::new(py, pref.iter().rev());
                            res.append(PyTuple::new(py, &[stack.to_object(py), empty_acc.clone_ref(py)]))?;
                        }
                        for (v, kids) in b.children.iter() {
                            for child in kids.values() {
                                pref.push(v.0.clone_ref(py));
                                dfs_upper(child, pref, res, py)?;
                                pref.pop();
                            }
                        }
                    }
                    Upper::Interface(i) => {
                        if let Some(empty_acc) = &i.empty {
                             let stack = PyList::new(py, pref.iter().rev());
                            res.append(PyTuple::new(py, &[stack.to_object(py), empty_acc.clone_ref(py)]))?;
                        }

                        if i.children.is_empty() && i.empty.is_none() {
                            let stack = PyList::new(py, pref.iter().rev());
                            res.append(PyTuple::new(py, &[stack.to_object(py), i.acc.clone_ref(py)]))?;
                        } else {
                            for (v, kids) in i.children.iter() {
                                for child in kids.values() {
                                    pref.push(v.0.clone_ref(py));
                                    dfs_lower(child, pref, &i.acc, res, py)?;
                                    pref.pop();
                                }
                            }
                        }
                    }
                }
                Ok(())
            }

            dfs_upper(&self.inner, &mut vec![], res, py)?;
            
            let reference_gss_module = py.import("gss_tester.implementations.reference_impl")?;
            let ref_gss_class = reference_gss_module.getattr("ReferenceGSS")?;
            let gss_instance = ref_gss_class.call1((res.to_object(py),))?;
            gss_instance.call_method0("to_stacks").map(|any| any.to_object(py))
        })
    }

    fn push(&self, value: PyObject) -> PyResult<Self> {
        if self.is_empty()? {
            return Ok(self.clone());
        }
        let new_inner = Python::with_gil(|py| {
            match &*self.inner {
                Upper::Interface(i) => {
                    let lower_node = Arc::new(Lower {
                        children: i.children.clone(),
                        empty: i.empty.is_some(),
                        max_depth: get_max_depth_lower(&i.children),
                    });
                    let mut new_children = Children::new();
                    new_children.insert(PyObjectWrapper(value), OrdMap::unit(lower_node.max_depth, lower_node));
                    let max_depth = get_max_depth_lower(&new_children);
                    Arc::new(Upper::Interface(Arc::new(Interface {
                        children: new_children,
                        acc: i.acc.clone_ref(py),
                        empty: None,
                        max_depth,
                    })))
                }
                Upper::Branch(_) => {
                    let mut new_children = Children::new();
                    new_children.insert(PyObjectWrapper(value), OrdMap::unit(self.inner.max_depth(), self.inner.clone()));
                    let max_depth = get_max_depth_upper(&new_children);
                    Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: new_children,
                        empty: None,
                        max_depth,
                    })))
                }
            }
        });
        Ok(LeveledGSS { inner: new_inner })
    }

    fn pop(&self) -> PyResult<Self> {
        // This is a simplified pop. A full implementation would need merging logic.
        Err(PyValueError::new_err("pop is not fully implemented in Rust yet"))
    }

    fn popn(&self, n: isize) -> PyResult<Self> {
        if n <= 0 {
            return Ok(self.clone());
        }
        // This is a simplified popn. A full implementation would need merging logic.
        Err(PyValueError::new_err("popn is not fully implemented in Rust yet"))
    }

    fn is_empty(&self) -> PyResult<bool> {
        Ok(match &*self.inner {
            Upper::Branch(b) => b.children.is_empty() && b.empty.is_none(),
            Upper::Interface(_) => false,
        })
    }

    fn isolate(&self, value: Option<PyObject>) -> PyResult<Self> {
        let new_inner = if let Some(val) = value {
            // Keep stacks with `value` at the top.
            match &*self.inner {
                Upper::Branch(b) => {
                    let filtered_children = b.children.get(&PyObjectWrapper(val.clone()))
                        .map(|kids| HashMap::unit(PyObjectWrapper(val), kids.clone()))
                        .unwrap_or_else(HashMap::new);
                    let max_depth = get_max_depth_upper(&filtered_children);
                    Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: filtered_children,
                        empty: None,
                        max_depth,
                    })))
                }
                Upper::Interface(i) => {
                    if let Some(kids) = i.children.get(&PyObjectWrapper(val.clone())) {
                        let filtered_children = HashMap::unit(PyObjectWrapper(val), kids.clone());
                        let max_depth = get_max_depth_lower(&filtered_children);
                         Arc::new(Upper::Interface(Arc::new(Interface {
                            children: filtered_children,
                            acc: i.acc.clone(),
                            empty: None,
                            max_depth,
                        })))
                    } else {
                        Arc::new(Upper::Branch(Arc::new(UpperBranch {
                            children: HashMap::new(),
                            empty: None,
                            max_depth: 0,
                        })))
                    }
                }
            }
        } else {
            // Keep only empty stacks.
            let empty_acc = match &*self.inner {
                Upper::Branch(b) => b.empty.clone(),
                Upper::Interface(i) => i.empty.clone(),
            };
            Arc::new(Upper::Branch(Arc::new(UpperBranch {
                children: HashMap::new(),
                empty: empty_acc,
                max_depth: 0,
            })))
        };
        Ok(LeveledGSS { inner: new_inner })
    }

    fn isolate_many(&self, values: &PySet) -> PyResult<Self> {
        let (new_empty, new_children_upper, new_children_lower) = Python::with_gil(|py| -> PyResult<_> {
            let mut new_empty = None;
            if values.contains(py.None())? {
                new_empty = match &*self.inner {
                    Upper::Branch(b) => b.empty.clone(),
                    Upper::Interface(i) => i.empty.clone(),
                };
            }

            let mut new_children_upper = HashMap::new();
            let mut new_children_lower = HashMap::new();

            match &*self.inner {
                Upper::Branch(b) => {
                    for (v, kids) in b.children.iter() {
                        if values.contains(&v.0)? {
                            new_children_upper.insert(v.clone(), kids.clone());
                        }
                    }
                }
                Upper::Interface(i) => {
                     for (v, kids) in i.children.iter() {
                        if values.contains(&v.0)? {
                            new_children_lower.insert(v.clone(), kids.clone());
                        }
                    }
                }
            }
            Ok((new_empty, new_children_upper, new_children_lower))
        })?;

        let new_inner = match &*self.inner {
            Upper::Branch(_) => {
                let max_depth = get_max_depth_upper(&new_children_upper);
                Arc::new(Upper::Branch(Arc::new(UpperBranch {
                    children: new_children_upper,
                    empty: new_empty,
                    max_depth,
                })))
            }
            Upper::Interface(i) => {
                if !new_children_lower.is_empty() {
                    let max_depth = get_max_depth_lower(&new_children_lower);
                    Arc::new(Upper::Interface(Arc::new(Interface {
                        children: new_children_lower,
                        acc: i.acc.clone(),
                        empty: new_empty,
                        max_depth,
                    })))
                } else {
                    Arc::new(Upper::Branch(Arc::new(UpperBranch {
                        children: HashMap::new(),
                        empty: new_empty,
                        max_depth: 0,
                    })))
                }
            }
        };

        Ok(LeveledGSS { inner: new_inner })
    }

    fn apply(&self, func: PyObject) -> PyResult<Self> {
        Err(PyValueError::new_err("apply is not implemented in Rust yet"))
    }

    fn prune(&self, predicate: PyObject) -> PyResult<Self> {
        Err(PyValueError::new_err("prune is not implemented in Rust yet"))
    }

    fn apply_and_prune(&self, mutator: PyObject) -> PyResult<Self> {
        Err(PyValueError::new_err("apply_and_prune is not implemented in Rust yet"))
    }

    fn merge(&self, other: &Self) -> PyResult<Self> {
        Err(PyValueError::new_err("merge is not implemented in Rust yet"))
    }

    fn peek(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let keys = self.inner.children()?;
            Ok(PySet::new(py, &keys)?.to_object(py))
        })
    }

    fn reduce_acc(&self) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            let mut unique_accs = std::collections::HashMap::new();
            let mut queue = VecDeque::new();
            queue.push_back(self.inner.clone());
            let mut visited = std::collections::HashSet::new();

            while let Some(node) = queue.pop_front() {
                let ptr = Arc::as_ptr(&node);
                if !visited.insert(ptr) {
                    continue;
                }

                match &*node {
                    Upper::Branch(b) => {
                        if let Some(acc) = &b.empty {
                            unique_accs.insert(acc.as_ref(py).as_ptr() as usize, acc.clone());
                        }
                        for kids in b.children.values() {
                            for child in kids.values() {
                                queue.push_back(child.clone());
                            }
                        }
                    }
                    Upper::Interface(i) => {
                        unique_accs.insert(i.acc.as_ref(py).as_ptr() as usize, i.acc.clone());
                        if let Some(acc) = &i.empty {
                            unique_accs.insert(acc.as_ref(py).as_ptr() as usize, acc.clone());
                        }
                    }
                }
            }

            let accs: Vec<PyObject> = unique_accs.values().cloned().collect();
            if accs.is_empty() {
                return Ok(None);
            }

            let mut it = accs.into_iter();
            let first = it.next().unwrap();
            let mut reduced = first;

            for next_acc in it {
                reduced = reduced.call_method1(py, "merge", (next_acc,))?;
            }

            Ok(Some(reduced))
        })
    }

    fn to_reference_impl(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stacks = self.to_stacks()?;
            let reference_gss_module = py.import("gss_tester.implementations.reference_impl")?;
            let ref_gss_class = reference_gss_module.getattr("ReferenceGSS")?;
            ref_gss_class.call_method1("from_stacks", (stacks,)).map(|any| any.to_object(py))
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

    #[classmethod]
    fn from_stacks(_cls: &PyType, stacks: &PyList) -> PyResult<Self> {
        Python::with_gil(|py| {
            let reference_gss_module = py.import("gss_tester.implementations.reference_impl")?;
            let ref_gss_class = reference_gss_module.getattr("ReferenceGSS")?;
            let gss_instance = ref_gss_class.call1((stacks,))?;
            let canonical_stacks: &PyList = gss_instance.getattr("_stacks")?.downcast()?;

            let mut empty_acc = None;
            let trie = PyDict::new(py);

            for item in canonical_stacks {
                let tuple: &PyTuple = item.downcast()?;
                let vals: &PyList = tuple.get_item(0)?.downcast()?;
                let acc: PyObject = tuple.get_item(1)?.to_object(py);

                if vals.is_empty() {
                    empty_acc = Some(acc);
                    continue;
                }

                let mut node = trie;
                let mut reversed_vals: Vec<PyObject> = vals.iter().map(|i| i.to_object(py)).collect();
                reversed_vals.reverse();

                for (i, v) in reversed_vals.iter().enumerate() {
                    let entry = if let Some(e) = node.get_item(v)? {
                        e.downcast()?
                    } else {
                        let new_entry = PyDict::new(py);
                        new_entry.set_item("end", py.None())?;
                        new_entry.set_item("sub", PyDict::new(py))?;
                        node.set_item(v, new_entry)?;
                        new_entry
                    };

                    if i == reversed_vals.len() - 1 {
                        entry.set_item("end", acc.clone_ref(py))?;
                    } else {
                        node = entry.get_item("sub")?.unwrap().downcast()?;
                    }
                }
            }

            // The `build` function from leveled_impl.py is complex to translate directly.
            // This is a placeholder for the trie-to-LeveledGSS conversion.
            // A full implementation would recursively build the Upper/Lower nodes from the trie.
            if !trie.is_empty() || empty_acc.is_some() {
                 return Err(PyValueError::new_err("from_stacks is not fully implemented in Rust yet: trie building is incomplete"));
            }

            Ok(LeveledGSS::_empty())
        })
    }
}

#[pymodule]
fn leveled_gss_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LeveledGSS>()?;
    Ok(())
}
