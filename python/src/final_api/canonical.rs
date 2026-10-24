//! Owned Module/Expr facade and Externs mapping view.
use super::*;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};

/// One declared extern contract on a Grammar or Module.
#[pyclass(name = "Extern", module = "glrmask", frozen, from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
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

    fn __repr__(&self) -> String {
        format!("Extern(name={:?}, kind={:?})", self.name, self.kind)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl From<glrmask::Extern> for PyExtern {
    fn from(e: glrmask::Extern) -> Self {
        let kind = match e.kind() {
            glrmask::ExternKind::Token => "token".to_string(),
            glrmask::ExternKind::Grammar => "grammar".to_string(),
            _ => "grammar".to_string(),
        };
        Self {
            name: e.name().to_string(),
            kind,
        }
    }
}

/// Read-only snapshot of a Module's declared externs. Implements collections.abc.Mapping semantics.
#[pyclass(name = "Externs", module = "glrmask", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub(super) struct PyExterns {
    pub(super) entries: Vec<PyExtern>,
    pub(super) unbound_entries: Option<Vec<PyExtern>>,
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

    /// The extern named `key`, or `default` if there is none.
    #[pyo3(signature = (key, default = None))]
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

    /// The extern names, in declaration order.
    fn keys(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// The externs, in declaration order.
    fn values(&self) -> Vec<PyExtern> {
        self.entries.clone()
    }

    /// `(name, extern)` pairs, in declaration order.
    fn items(&self) -> Vec<(String, PyExtern)> {
        self.entries
            .iter()
            .map(|e| (e.name.clone(), e.clone()))
            .collect()
    }

    fn __repr__(&self) -> String {
        let pairs: Vec<String> = self
            .entries
            .iter()
            .map(|e| format!("{:?}: {:?}", e.name, e))
            .collect();
        format!("Externs({{{}}})", pairs.join(", "))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

/// Immutable union or concatenation of independently scoped Grammar, Module or Expr snapshots.
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
            "plain strings are not accepted as Expr parts; use Expr.literal(...)",
        ));
    }
    if let Ok(m) = value.extract::<PyRef<'_, PyModule>>() {
        let mod_inner = m.snapshot();
        let unbound = mod_inner.externs().unbound();
        if !unbound.is_empty() {
            let names: Vec<String> = unbound.iter().map(|e| e.name().to_string()).collect();
            return Err(PyValueError::new_err(format!(
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
            return Err(PyValueError::new_err(format!(
                "cannot create Expr: part has unbound externs [{}]",
                names
                    .iter()
                    .map(|n| format!("\"{n}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        return Ok(glrmask::Expr::from(m));
    }
    if let Ok(e) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(e.inner.clone());
    }
    if let Ok(et) = value.extract::<PyRef<'_, PyTokenSet>>() {
        return Ok(glrmask::Expr::from(et.inner.clone()));
    }
    Err(PyTypeError::new_err(
        "Expr parts must be Grammar, Module, Expr, or TokenSet",
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
    /// Match exact bytes (str is encoded as UTF-8). No whitespace is inserted between parts.
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

    /// Unordered union. Empty choice matches nothing. One-part choice is that part.
    #[staticmethod]
    #[pyo3(signature = (*parts))]
    fn choice(parts: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let items = collect_parts(parts)?;
        Ok(Self {
            inner: glrmask::Expr::choice(items),
        })
    }

    /// Concatenation; model tokens may cross operand boundaries. Empty sequence is epsilon.
    #[staticmethod]
    #[pyo3(signature = (*parts))]
    fn sequence(parts: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let items = collect_parts(parts)?;
        Ok(Self {
            inner: glrmask::Expr::sequence(items),
        })
    }
}

/// Owned mutable assembly. Holds a source Grammar, local settings, bindings and retained compilation work.
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

fn extract_binding(value: &Bound<'_, PyAny>) -> PyResult<glrmask::Binding> {
    if let Ok(m) = value.extract::<PyRef<'_, PyModule>>() {
        return Ok(glrmask::Binding::from(m.snapshot()));
    }
    if let Ok(g) = value.extract::<PyRef<'_, PyGrammar>>() {
        return Ok(glrmask::Binding::from(&g.inner));
    }
    if let Ok(e) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(glrmask::Binding::from(e.inner.clone()));
    }
    if let Ok(et) = value.extract::<PyRef<'_, PyTokenSet>>() {
        return Ok(glrmask::Binding::from(et.inner.clone()));
    }
    Err(PyTypeError::new_err(
        "Module.bind accepts Grammar, Module, Expr, or TokenSet",
    ))
}

pub(super) fn bind_grammar(
    grammar: &mut glrmask::Grammar,
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if let Ok(g) = value.extract::<PyRef<'_, PyGrammar>>() {
        return grammar.bind(name, g.inner.clone()).map_err(api_error);
    }
    if let Ok(e) = value.extract::<PyRef<'_, PyExpr>>() {
        return grammar.bind(name, e.inner.clone()).map_err(api_error);
    }
    if let Ok(t) = value.extract::<PyRef<'_, PyTokenSet>>() {
        return grammar.bind(name, t.inner.clone()).map_err(api_error);
    }
    Err(PyTypeError::new_err(
        "Grammar.bind accepts Grammar, Expr, or TokenSet",
    ))
}

pub(super) fn compile_options(
    mask_strategy: Option<PyRef<'_, PyMaskStrategy>>,
    boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
    parser_backend: Option<PyRef<'_, PyParserBackend>>,
    merge_equivalent_tokens: bool,
) -> glrmask::CompileOptions {
    let mask_strategy = mask_strategy
        .as_deref()
        .copied()
        .unwrap_or(PyMaskStrategy::AUTO);
    let boundary_mask_strategy = boundary_mask_strategy
        .as_deref()
        .copied()
        .unwrap_or(PyBoundaryMaskStrategy::AUTO);
    let parser_backend = parser_backend
        .as_deref()
        .copied()
        .unwrap_or(PyParserBackend::LR_TABLE);
    glrmask::CompileOptions::default()
        .mask_strategy(mask_strategy.into())
        .boundary_mask_strategy(boundary_mask_strategy.into())
        .parser_backend(parser_backend.into())
        .merge_equivalent_tokens(merge_equivalent_tokens)
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

    #[pyo3(signature = (*args, **kwargs))]
    fn bind(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let has_args = !args.is_empty();
        let has_kwargs = kwargs.map_or(false, |kw| !kw.is_empty());

        if has_args && has_kwargs {
            return Err(PyTypeError::new_err(
                "cannot mix positional and keyword arguments in Module.bind",
            ));
        }
        if !has_args && !has_kwargs {
            return Err(PyTypeError::new_err(
                "Module.bind expects either positional (name, value) or keyword bindings",
            ));
        }

        if has_args {
            if args.len() != 2 {
                return Err(PyTypeError::new_err(format!(
                    "Module.bind positional form takes exactly 2 arguments (name, value), got {}",
                    args.len()
                )));
            }
            let name: String = args.get_item(0)?.extract()?;
            let value = args.get_item(1)?;
            let binding = extract_binding(&value)?;
            let mut target = slf.try_borrow_mut()?;
            let mut next = target.snapshot();
            let next = py
                .detach(move || {
                    next.bind(&name, binding)?;
                    Ok::<_, glrmask::Error>(next)
                })
                .map_err(api_error)?;
            target.replace(py, next);
            Ok(())
        } else {
            let kwargs = kwargs.unwrap();
            let mut bindings = Vec::with_capacity(kwargs.len());
            for (key, val) in kwargs.iter() {
                let name: String = key
                    .extract()
                    .map_err(|_| PyTypeError::new_err("binding names must be strings"))?;
                let binding = extract_binding(&val)?;
                bindings.push((name, binding));
            }
            let mut target = slf.try_borrow_mut()?;
            let mut next = target.snapshot();
            let next = py
                .detach(move || {
                    for (name, binding) in bindings {
                        next.bind(&name, binding)?;
                    }
                    Ok::<_, glrmask::Error>(next)
                })
                .map_err(api_error)?;
            target.replace(py, next);
            Ok(())
        }
    }

    #[pyo3(signature = (vocab, *, mask_strategy = None, boundary_mask_strategy = None, parser_backend = None, merge_equivalent_tokens = false))]
    fn precompile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<PyRef<'_, PyMaskStrategy>>,
        boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
        parser_backend: Option<PyRef<'_, PyParserBackend>>,
        merge_equivalent_tokens: bool,
    ) -> PyResult<()> {
        let vocab = vocab.inner.clone();
        let options = compile_options(mask_strategy, boundary_mask_strategy, parser_backend, merge_equivalent_tokens);
        let mut target = slf.try_borrow_mut()?;
        let mut next = target.snapshot();
        let next = py
            .detach(move || {
                next.precompile_with(&vocab, options)?;
                Ok::<_, glrmask::Error>(next)
            })
            .map_err(api_error)?;
        target.replace(slf.py(), next);
        Ok(())
    }

    #[pyo3(signature = (vocab, *, mask_strategy = None, boundary_mask_strategy = None, parser_backend = None, merge_equivalent_tokens = false))]
    fn compile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<PyRef<'_, PyMaskStrategy>>,
        boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
        parser_backend: Option<PyRef<'_, PyParserBackend>>,
        merge_equivalent_tokens: bool,
    ) -> PyResult<PyConstraint> {
        let snapshot = slf.try_borrow()?.snapshot();
        let vocab = vocab.inner.clone();
        let options = compile_options(mask_strategy, boundary_mask_strategy, parser_backend, merge_equivalent_tokens);
        let constraint = py
            .detach(move || snapshot.compile_with(&vocab, options))
            .map_err(api_error)?;
        Ok(super::constraint(constraint))
    }

    #[getter]
    fn externs(slf: &Bound<'_, Self>) -> PyResult<PyExterns> {
        let snapshot = slf.borrow().snapshot();
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
        let bytes = slf.py().detach(move || snapshot.save());
        Ok(PyBytes::new(slf.py(), &bytes))
    }

    #[staticmethod]
    #[pyo3(signature = (data, vocab = None))]
    fn load(py: Python<'_>, data: &[u8], vocab: Option<&PyVocab>) -> PyResult<Self> {
        let data = data.to_vec();
        let vocab = vocab.map(|v| v.inner.clone());
        py.detach(move || glrmask::Module::load(&data))
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
