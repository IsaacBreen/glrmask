//! Owned Module/Expr facade and Externs mapping view.
use super::*;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use crate::glrmask_error;
use pyo3::types::PyString;

fn extern_repr(py: Python<'_>, name: &str, kind: &str) -> PyResult<String> {
    let name = PyString::new(py, name).repr()?.to_string();
    let kind = PyString::new(py, kind).repr()?.to_string();
    Ok(format!("Extern(name={name}, kind={kind})"))
}

/// A named placeholder in a grammar.
#[pyclass(name = "Extern", module = "glrmask", frozen, from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct PyExtern {
    pub(super) name: String,
    pub(super) kind: String,
}

#[pymethods]
impl PyExtern {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        extern_repr(py, &self.name, &self.kind)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match other.extract::<PyRef<'_, Self>>() {
            Ok(other) => Ok(pyo3::types::PyBool::new(py, self == &*other)
                .to_owned()
                .into_any()
                .unbind()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }
}

impl From<&glrmask::Extern> for PyExtern {
    fn from(e: &glrmask::Extern) -> Self {
        let kind = match e.kind() {
            glrmask::ExternKind::Token => "token".to_string(),
            glrmask::ExternKind::Grammar => "grammar".to_string(),
            _ => unreachable!("every extern kind has a binding name"),
        };
        Self {
            name: e.name().to_string(),
            kind,
        }
    }
}

/// A read-only mapping of extern names to `Extern`.
#[pyclass(name = "Externs", module = "glrmask", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub(super) struct PyExterns {
    pub(super) entries: Vec<PyExtern>,
    pub(super) unbound_entries: Option<Vec<PyExtern>>,
}

impl PyExterns {
    fn unbound_records(&self) -> &[PyExtern] {
        self.unbound_entries.as_deref().unwrap_or(&self.entries)
    }
}

#[pymethods]
impl PyExterns {
    #[getter]
    fn unbound(&self) -> Self {
        match &self.unbound_entries {
            Some(entries) => Self {
                entries: entries.clone(),
                unbound_entries: None,
            },
            None => self.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.entries.len()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.entries.iter().any(|e| e.name == key)
    }

    fn __getitem__(&self, key: &str) -> PyResult<PyExtern> {
        self.entries
            .iter()
            .find(|e| e.name == key)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    #[pyo3(signature = (key, default = None, /))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        if let Some(e) = self.entries.iter().find(|e| e.name == key) {
            Ok(e.clone().into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let names: Vec<String> = slf.borrow().entries.iter().map(|e| e.name.clone()).collect();
        let py_list = pyo3::types::PyList::new(py, names)?;
        Ok(py_list.call_method0("__iter__")?.unbind())
    }

    fn keys(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf.py().import("collections.abc")?.getattr("KeysView")?.call1((slf,))?.unbind())
    }

    fn values(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf.py().import("collections.abc")?.getattr("ValuesView")?.call1((slf,))?.unbind())
    }

    fn items(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf.py().import("collections.abc")?.getattr("ItemsView")?.call1((slf,))?.unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut pairs: Vec<String> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let key = PyString::new(py, &entry.name).repr()?.to_string();
            pairs.push(format!("{key}: {}", extern_repr(py, &entry.name, &entry.kind)?));
        }
        let names = self.unbound_records().iter().map(|e| {
            Ok(PyString::new(py, &e.name).repr()?.to_string())
        }).collect::<PyResult<Vec<_>>>()?;
        Ok(format!("Externs({{{}}}, unbound=[{}])", pairs.join(", "), names.join(", ")))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match other.extract::<PyRef<'_, Self>>() {
            Ok(other) => Ok(pyo3::types::PyBool::new(py, self.entries == other.entries && self.unbound_records() == other.unbound_records())
                .to_owned()
                .into_any()
                .unbind()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }
}

/// A grammar expression.
#[pyclass(name = "Expr", module = "glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyExpr {
    pub(super) inner: glrmask::Expr,
}

impl Drop for PyExpr {
    fn drop(&mut self) {
        let inner = std::mem::replace(
            &mut self.inner,
            glrmask::Expr::sequence(Vec::<glrmask::Expr>::new()),
        );
        let _ = Python::try_attach(move |py| py.detach(move || drop(inner)));
    }
}

pub(super) fn extract_expr_part(value: &Bound<'_, PyAny>) -> PyResult<glrmask::Expr> {
    if value.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyTypeError::new_err(
            "plain strings are not Expr operands; wrap them in a literal expression",
        ));
    }
    if let Ok(m) = value.extract::<PyRef<'_, PyModule>>() {
        let mod_inner = m.snapshot();
        let unbound = mod_inner.externs().unbound();
        if !unbound.is_empty() {
            let names: Vec<String> = unbound.iter().map(|e| e.name().to_string()).collect();
            return Err(glrmask_error("invalid_input", format!(
                "cannot create Expr: part has unbound externs [{}]",
                names
                    .iter()
                    .map(|n| format!("\"{n}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        return Ok(glrmask::Expr::from(mod_inner));
    }
    if let Ok(g) = value.extract::<PyRef<'_, PyGrammar>>() {
        let m = glrmask::Module::new(&g.inner);
        let unbound = m.externs().unbound();
        if !unbound.is_empty() {
            let names: Vec<String> = unbound.iter().map(|e| e.name().to_string()).collect();
            return Err(glrmask_error("invalid_input", format!(
                "cannot create Expr: part has unbound externs [{}]",
                names
                    .iter()
                    .map(|n| format!("\"{n}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        return Ok(glrmask::Expr::from(&g.inner));
    }
    if let Ok(e) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(e.inner.clone());
    }
    if let Ok(et) = value.extract::<PyRef<'_, PyTokenSet>>() {
        return Ok(glrmask::Expr::from(et.inner.clone()));
    }
    Err(PyTypeError::new_err(
        "Expr parts must be Grammar, Module, Expr or TokenSet",
    ))
}

fn collect_parts(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Vec<glrmask::Expr>> {
    let mut parts = Vec::new();
    let items: Vec<Bound<'_, PyAny>> = if args.len() == 1 {
        let first = args.get_item(0)?;
        if first.is_instance_of::<pyo3::types::PyList>()
            || first.is_instance_of::<pyo3::types::PyTuple>()
        {
            first.try_iter()?.collect::<PyResult<Vec<_>>>()?
        } else {
            vec![first]
        }
    } else {
        args.iter().collect()
    };

    for item in items {
        parts.push(extract_expr_part(&item)?);
    }
    Ok(parts)
}

#[pymethods]
impl PyExpr {
    /// Matches `value` exactly.
    #[staticmethod]
    #[pyo3(signature = (value, /))]
    fn literal(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bytes = if let Ok(s) = value.extract::<String>() {
            s.into_bytes()
        } else if let Ok(b) = value.extract::<Vec<u8>>() {
            b
        } else {
            return Err(PyTypeError::new_err("Expr.literal requires str or bytes"));
        };
        Ok(Self {
            inner: glrmask::Expr::literal(bytes),
        })
    }

    /// Matches any one of `parts`.
    #[staticmethod]
    #[pyo3(signature = (*parts))]
    fn choice(parts: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let items = collect_parts(parts)?;
        Ok(Self {
            inner: glrmask::Expr::choice(items),
        })
    }

    /// Matches `parts` in order.
    #[staticmethod]
    #[pyo3(signature = (*parts))]
    fn sequence(parts: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let items = collect_parts(parts)?;
        Ok(Self {
            inner: glrmask::Expr::sequence(items),
        })
    }
}

/// A grammar you can compose with others and partly compile before its externs are bound.
#[pyclass(name = "Module", module = "glrmask", from_py_object)]
#[derive(Clone)]
pub(super) struct PyModule {
    pub(super) inner: Option<glrmask::Module>,
}

impl PyModule {
    pub(super) fn from_inner(inner: glrmask::Module) -> Self {
        Self { inner: Some(inner) }
    }
    fn snapshot(&self) -> glrmask::Module {
        self.inner.as_ref().expect("live Module").clone()
    }
    fn replace(&mut self, py: Python<'_>, next: glrmask::Module) {
        let previous = self.inner.replace(next);
        py.detach(move || drop(previous));
    }
}

impl Drop for PyModule {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let _ = Python::try_attach(move |py| py.detach(move || drop(inner)));
        }
    }
}

enum ModuleBinding {
    Module(glrmask::Module),
    Grammar(glrmask::Grammar),
    Expr(glrmask::Expr),
    TokenSet(glrmask::TokenSet),
}

impl ModuleBinding {
    fn bind(self, module: &mut glrmask::Module, name: &str) -> glrmask::Result<()> {
        match self {
            Self::Module(value) => module.bind(name, value),
            Self::Grammar(value) => module.bind(name, value),
            Self::Expr(value) => module.bind(name, value),
            Self::TokenSet(value) => module.bind(name, value),
        }
    }
}

fn extract_binding(value: &Bound<'_, PyAny>) -> PyResult<ModuleBinding> {
    if let Ok(m) = value.extract::<PyRef<'_, PyModule>>() {
        return Ok(ModuleBinding::Module(m.snapshot()));
    }
    if let Ok(g) = value.extract::<PyRef<'_, PyGrammar>>() {
        return Ok(ModuleBinding::Grammar(g.inner.clone()));
    }
    if let Ok(e) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(ModuleBinding::Expr(e.inner.clone()));
    }
    if let Ok(et) = value.extract::<PyRef<'_, PyTokenSet>>() {
        return Ok(ModuleBinding::TokenSet(et.inner.clone()));
    }
    Err(PyTypeError::new_err(
        "Module.bind accepts a Grammar, Module, Expr, or TokenSet",
    ))
}

pub(super) fn compile_options(mask_strategy: super::PyMaskStrategy) -> glrmask::CompileOptions {
    glrmask::CompileOptions::default().mask_strategy(glrmask::MaskStrategy::from(mask_strategy))
}

#[pymethods]
impl PyModule {
    #[new]
    #[pyo3(signature = (source, /))]
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let grammar = source
            .extract::<PyRef<'_, PyGrammar>>()
            .map_err(|_| PyTypeError::new_err("Module expects Grammar"))?;
        Ok(Self::from_inner(glrmask::Module::new(&grammar.inner)))
    }

    /// Binds an extern by name.
    #[pyo3(signature = (name, value, /))]
    fn bind(slf: &Bound<'_, Self>, py: Python<'_>, name: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let binding = extract_binding(value)?;
        let mut target = slf.try_borrow_mut()?;
        let mut next = target.snapshot();
        let next = super::super::catch_python_panic(|| py.detach(move || {
            binding.bind(&mut next, &name)?;
            Ok::<_, glrmask::Error>(next)
        }))?.map_err(api_error)?;
        target.replace(py, next);
        Ok(())
    }

    /// Compiles what it can now, even with externs unbound. Further compilation must use the same `vocab`.
    #[pyo3(signature = (vocab, *, mask_strategy = super::PyMaskStrategy::AUTO), text_signature = "($self, vocab, *, mask_strategy=...)")]
    fn precompile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: super::PyMaskStrategy,
    ) -> PyResult<()> {
        let vocab = vocab.inner.clone();
        let options = compile_options(mask_strategy);
        let mut target = slf.try_borrow_mut()?;
        let mut next = target.snapshot();
        let next = super::super::catch_python_panic(|| py
            .detach(move || {
                next.precompile_with(&vocab, options)?;
                Ok::<_, glrmask::Error>(next)
            }))?
            .map_err(api_error)?;
        target.replace(slf.py(), next);
        Ok(())
    }

    /// Finishes compiling the module into a constraint.
    #[pyo3(signature = (vocab, *, mask_strategy = super::PyMaskStrategy::AUTO), text_signature = "($self, vocab, *, mask_strategy=...)")]
    fn compile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: super::PyMaskStrategy,
    ) -> PyResult<PyConstraint> {
        let snapshot = slf.try_borrow()?.snapshot();
        let vocab = vocab.inner.clone();
        let options = compile_options(mask_strategy);
        let constraint = super::super::catch_python_panic(|| py
            .detach(move || snapshot.compile_with(&vocab, options)))?
            .map_err(api_error)?;
        Ok(super::constraint(constraint))
    }

    #[getter]
    fn externs(slf: &Bound<'_, Self>) -> PyResult<PyExterns> {
        let snapshot = slf.try_borrow()?.snapshot();
        let all_externs: Vec<PyExtern> = snapshot.externs().iter().map(PyExtern::from).collect();
        let unbound_externs: Vec<PyExtern> = snapshot
            .externs()
            .unbound()
            .iter()
            .map(PyExtern::from)
            .collect();
        Ok(PyExterns {
            entries: all_externs,
            unbound_entries: Some(unbound_externs),
        })
    }

    fn copy(&self) -> Self {
        Self::from_inner(self.snapshot())
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __deepcopy__(slf: &Bound<'_, Self>, memo: &Bound<'_, PyDict>) -> PyResult<Py<Self>> {
        let copy = Py::new(slf.py(), slf.try_borrow()?.copy())?;
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        Ok(copy)
    }

    fn save<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyBytes>> {
        let snapshot = slf.try_borrow()?.snapshot();
        let bytes = super::super::catch_python_panic(|| slf.py().detach(move || snapshot.save()))?;
        Ok(PyBytes::new(slf.py(), &bytes))
    }

    /// May fail if saved by a different GLRMask version.
    #[staticmethod]
    #[pyo3(signature = (data, /))]
    fn load(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        let data = data.to_vec();
        super::super::catch_python_panic(|| py.detach(move || glrmask::Module::load(&data)))?
            .map(Self::from_inner)
            .map_err(api_error)
    }
}

pub(super) fn register(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_class::<PyModule>()?;
    module.add_class::<PyExpr>()?;
    module.add_class::<PyExtern>()?;
    module.add_class::<PyExterns>()?;
    Ok(())
}
