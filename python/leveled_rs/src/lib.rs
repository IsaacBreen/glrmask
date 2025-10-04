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

    fn popn(&self, n: isize) -> PyResult<Self> {
        if n <= 0 {
            return Ok(self.clone());
        }
        if self.is_empty()? {
            return Ok(self.clone());
        }

        Python::with_gil(|py| {
            let mut memo_upper: HashMap<usize, Arc<Upper>> = HashMap::new();
            let mut memo_lower: HashMap<usize, Arc<Lower>> = HashMap::new();

            fn popn_lower(py: Python, node: &Arc<Lower>, k: isize, memo_lower: &mut HashMap<usize, Arc<Lower>>) -> PyResult<Arc<Lower>> {
                if k == 0 {
                    return Ok(node.clone());
                }
                let key = Arc::as_ptr(node) as usize;
                if let Some(cached) = memo_lower.get(&key) {
                    return Ok(cached.clone());
                }

                let all_children: Vec<_> = node.children.values().flat_map(|kids| kids.values()).cloned().collect();
                if all_children.is_empty() {
                    let res = Arc::new(Lower { children: HashMap::new(), empty: false, max_depth: 0 });
                    memo_lower.insert(key, res.clone());
                    return Ok(res);
                }

                let mut popped_children = Vec::new();
                for child in all_children {
                    popped_children.push(popn_lower(py, &child, k - 1, memo_lower)?);
                }

                let mut it = popped_children.into_iter();
                let first = it.next().unwrap();
                let res = it.try_fold(first, |acc, next| LeveledGSS::merge_lower(py, &acc, &next))?;
                memo_lower.insert(key, res.clone());
                Ok(res)
            }

            fn popn_upper(py: Python, node: &Arc<Upper>, k: isize, memo_upper: &mut HashMap<usize, Arc<Upper>>, memo_lower: &mut HashMap<usize, Arc<Lower>>) -> PyResult<Arc<Upper>> {
                if k == 0 {
                    return Ok(node.clone());
                }
                let key = Arc::as_ptr(node) as usize;
                if let Some(cached) = memo_upper.get(&key) {
                    return Ok(cached.clone());
                }

                let res = match &**node {
                    Upper::Branch(b) => {
                        let all_children: Vec<_> = b.children.values().flat_map(|kids| kids.values()).cloned().collect();
                        if all_children.is_empty() {
                            return Ok(LeveledGSS::_empty().inner);
                        }
                        let mut popped_children = Vec::new();
                        for child in all_children {
                            popped_children.push(popn_upper(py, &child, k - 1, memo_upper, memo_lower)?);
                        }
                        let mut it = popped_children.into_iter();
                        let first = it.next().unwrap();
                        let merged = it.try_fold(first, |acc, next| LeveledGSS::merge_upper(py, &acc, &next))?;
                        LeveledGSS::try_promote(py, &merged)?
                    }
                    Upper::Interface(i) => {
                        let all_children: Vec<_> = i.children.values().flat_map(|kids| kids.values()).cloned().collect();
                         if all_children.is_empty() {
                            return Ok(LeveledGSS::_empty().inner);
                        }
                        let mut popped_children = Vec::new();
                        for child in all_children {
                            popped_children.push(popn_lower(py, &child, k - 1, memo_lower)?);
                        }
                        let mut it = popped_children.into_iter();
                        let first = it.next().unwrap();
                        let merged = it.try_fold(first, |acc, next| LeveledGSS::merge_lower(py, &acc, &next))?;

                        let new_empty = if merged.empty { Some(i.acc.clone()) } else { None };
                        if merged.children.is_empty() && new_empty.is_none() {
                            LeveledGSS::_empty().inner
                        } else {
                            let max_depth = get_max_depth_lower(&merged.children);
                            Arc::new(Upper::Interface(Arc::new(Interface {
                                children: merged.children.clone(),
                                acc: i.acc.clone(),
                                empty: new_empty,
                                max_depth,
                            })))
                        }
                    }
                };

                memo_upper.insert(key, res.clone());
                Ok(res)
            }

            let new_inner = popn_upper(py, &self.inner, n, &mut memo_upper, &mut memo_lower)?;
            Ok(LeveledGSS { inner: new_inner })
        })
    }

    fn pop(&self) -> PyResult<Self> {
        self.popn(1)
    }

    fn is_empty(&self) -> PyResult<bool> {
        Ok(match &*self.inner {
            Upper::Branch(b) => b.children.is_empty() && b.empty.is_none(),
            Upper::Interface(_) => false,
        })
    }

    fn apply(&self, func: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut memo: HashMap<usize, Arc<Upper>> = HashMap::new();

            fn transform(py: Python, node: &Arc<Upper>, func: &PyObject, memo: &mut HashMap<usize, Arc<Upper>>) -> PyResult<Arc<Upper>> {
                let key = Arc::as_ptr(node) as usize;
                if let Some(cached) = memo.get(&key) {
                    return Ok(cached.clone());
                }

                let res = match &**node {
                    Upper::Interface(i) => {
                        let new_acc = func.call1(py, (i.acc.clone(),))?;
                        let new_empty = if let Some(empty) = &i.empty {
                            Some(func.call1(py, (empty.clone(),))?)
                        } else {
                            None
                        };
                        let new_i = Arc::new(Upper::Interface(Arc::new(Interface {
                            children: i.children.clone(),
                            acc: new_acc,
                            empty: new_empty,
                            max_depth: i.max_depth,
                        })));
                        LeveledGSS::try_promote(py, &new_i)?
                    }
                    Upper::Branch(b) => {
                        let new_empty = if let Some(empty) = &b.empty {
                            Some(func.call1(py, (empty.clone(),))?)
                        } else {
                            None
                        };
                        let mut new_children = Children::new();
                        for (v, kids) in b.children.iter() {
                            let mut new_kids_for_v = OrdMap::new();
                            for child in kids.values() {
                                let new_child = transform(py, child, func, memo)?;
                                new_kids_for_v.insert(new_child.max_depth(), new_child);
                            }
                            new_children.insert(v.clone(), new_kids_for_v);
                        }
                        let max_depth = get_max_depth_upper(&new_children);
                        let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                            children: new_children,
                            empty: new_empty,
                            max_depth,
                        })));
                        LeveledGSS::try_promote(py, &new_b)?
                    }
                };
                memo.insert(key, res.clone());
                Ok(res)
            }

            let new_inner = transform(py, &self.inner, &func, &mut memo)?;
            Ok(LeveledGSS { inner: new_inner })
        })
    }

    fn prune(&self, predicate: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut memo: HashMap<usize, Option<Arc<Upper>>> = HashMap::new();

            fn transform(py: Python, node: &Arc<Upper>, predicate: &PyObject, memo: &mut HashMap<usize, Option<Arc<Upper>>>) -> PyResult<Option<Arc<Upper>>> {
                let key = Arc::as_ptr(node) as usize;
                if let Some(cached) = memo.get(&key) {
                    return Ok(cached.clone());
                }

                let res = match &**node {
                    Upper::Interface(i) => {
                        let keep_acc = predicate.call1(py, (i.acc.clone(),))?.is_true(py)?;
                        let keep_empty = if let Some(empty) = &i.empty {
                            predicate.call1(py, (empty.clone(),))?.is_true(py)?
                        } else {
                            false
                        };

                        if !keep_acc && !keep_empty {
                            None
                        } else if !keep_acc && keep_empty {
                            let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                                children: HashMap::new(),
                                empty: i.empty.clone(),
                                max_depth: 0,
                            })));
                            Some(LeveledGSS::try_promote(py, &new_b)?)
                        } else { // keep_acc is true
                            let new_empty = if keep_empty { i.empty.clone() } else { None };
                            let new_i = Arc::new(Upper::Interface(Arc::new(Interface {
                                children: i.children.clone(),
                                acc: i.acc.clone(),
                                empty: new_empty,
                                max_depth: i.max_depth,
                            })));
                            Some(LeveledGSS::try_promote(py, &new_i)?)
                        }
                    }
                    Upper::Branch(b) => {
                        let new_empty = if let Some(empty) = &b.empty {
                            if predicate.call1(py, (empty.clone(),))?.is_true(py)? {
                                Some(empty.clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let mut new_children = Children::new();
                        for (v, kids) in b.children.iter() {
                            let mut new_kids_for_v = OrdMap::new();
                            for child in kids.values() {
                                if let Some(new_child) = transform(py, child, predicate, memo)? {
                                    new_kids_for_v.insert(new_child.max_depth(), new_child);
                                }
                            }
                            if !new_kids_for_v.is_empty() {
                                new_children.insert(v.clone(), new_kids_for_v);
                            }
                        }

                        if new_children.is_empty() && new_empty.is_none() {
                            None
                        } else {
                            let max_depth = get_max_depth_upper(&new_children);
                            let new_b = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                                children: new_children,
                                empty: new_empty,
                                max_depth,
                            })));
                            Some(LeveledGSS::try_promote(py, &new_b)?)
                        }
                    }
                };
                memo.insert(key, res.clone());
                Ok(res)
            }

            let new_inner = transform(py, &self.inner, &predicate, &mut memo)?;
            Ok(new_inner.map_or_else(LeveledGSS::_empty, |inner| LeveledGSS { inner }))
        })
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

    fn apply_and_prune(&self, mutator: PyObject) -> PyResult<Self> {
        // The default implementation in Python is prune + apply.
        // A single-pass version is an optimization. For now, let's use the composed version.
        let pruned = self.prune(mutator.clone())?;
        pruned.apply(mutator)
    }

    fn merge(&self, other: &Self) -> PyResult<Self> {
        Python::with_gil(|py| {
            let merged_inner = LeveledGSS::merge_upper(py, &self.inner, &other.inner)?;
            Ok(LeveledGSS { inner: merged_inner })
        })
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

            fn build_lower(py: Python, d: &PyDict) -> PyResult<Arc<Lower>> {
                let mut l_children = Children::new();
                for (v_obj, e_obj) in d.iter() {
                    let e: &PyDict = e_obj.downcast()?;
                    let sub_l_dict: &PyDict = e.get_item("sub")?.unwrap().downcast()?;
                    let has_end = !e.get_item("end")?.unwrap().is_none();

                    let sub_lower = if sub_l_dict.is_empty() {
                        Arc::new(Lower { children: HashMap::new(), empty: false, max_depth: 0 })
                    } else {
                        build_lower(py, sub_l_dict)?
                    };

                    let node_for_v = Arc::new(Lower {
                        children: sub_lower.children.clone(),
                        empty: has_end,
                        max_depth: get_max_depth_lower(&sub_lower.children),
                    });
                    l_children.insert(PyObjectWrapper(v_obj.to_object(py)), OrdMap::unit(node_for_v.max_depth, node_for_v));
                }
                let max_depth = get_max_depth_lower(&l_children);
                Ok(Arc::new(Lower { children: l_children, empty: false, max_depth }))
            }

            fn build(py: Python, d: &PyDict, root_empty: Option<PyObject>) -> PyResult<Arc<Upper>> {
                let mut children = Children::new();
                let mut all_child_nodes = Vec::new();

                for (v_obj, e_obj) in d.iter() {
                    let e: &PyDict = e_obj.downcast()?;
                    let mut nodes_for_v = Vec::new();
                    let end_acc = e.get_item("end")?.unwrap();
                    let sub_dict: &PyDict = e.get_item("sub")?.unwrap().downcast()?;

                    if !end_acc.is_none() {
                        let leaf = Arc::new(Upper::Branch(Arc::new(UpperBranch {
                            children: HashMap::new(),
                            empty: Some(end_acc.to_object(py)),
                            max_depth: 0,
                        })));
                        nodes_for_v.push(LeveledGSS::try_promote(py, &leaf)?);
                    }
                    if !sub_dict.is_empty() {
                        nodes_for_v.push(build(py, sub_dict, None)?);
                    }
                    if !nodes_for_v.is_empty() {
                        let mut kids_map = OrdMap::new();
                        for n in &nodes_for_v {
                            kids_map.insert(n.max_depth(), n.clone());
                        }
                        children.insert(PyObjectWrapper(v_obj.to_object(py)), kids_map);
                        all_child_nodes.extend(nodes_for_v);
                    }
                }

                let all_interfaces = all_child_nodes.iter().all(|c| matches!(&**c, Upper::Interface(_)));

                if all_interfaces {
                    let mut accs = std::collections::HashSet::new();
                    for node in &all_child_nodes {
                        if let Upper::Interface(i) = &**node {
                            accs.insert(PyObjectWrapper(i.acc.clone()));
                            if let Some(empty) = &i.empty {
                                accs.insert(PyObjectWrapper(empty.clone()));
                            }
                        }
                    }
                    if let Some(empty) = &root_empty {
                        accs.insert(PyObjectWrapper(empty.clone()));
                    }

                    if accs.len() <= 1 {
                        if let Some(the_acc_wrapper) = accs.into_iter().next() {
                            let lower_tree = build_lower(py, d)?;
                            let max_depth = get_max_depth_lower(&lower_tree.children);
                            return Ok(Arc::new(Upper::Interface(Arc::new(Interface {
                                children: lower_tree.children.clone(),
                                acc: the_acc_wrapper.0,
                                empty: root_empty,
                                max_depth,
                            }))));
                        } else {
                             return Ok(LeveledGSS::_empty().inner);
                        }
                    }
                }

                let max_depth = get_max_depth_upper(&children);
                Ok(Arc::new(Upper::Branch(Arc::new(UpperBranch {
                    children,
                    empty: root_empty,
                    max_depth,
                }))))
            }

            let inner = build(py, trie, empty_acc)?;
            Ok(LeveledGSS { inner })
        })
    }
}

#[pymodule]
fn leveled_gss_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LeveledGSS>()?;
    Ok(())
}
