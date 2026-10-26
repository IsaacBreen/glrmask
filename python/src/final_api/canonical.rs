//! Owned Module/Expr facade and Externs mapping view.
use super::*;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::types::PyString;

fn extern_repr(py: Python<'_>, name: &str, kind: &str) -> PyResult<String> {
    let name = PyString::new(py, name).repr()?.to_string();
    let kind = PyString::new(py, kind).repr()?.to_string();
    Ok(format!("Extern(name={name}, kind={kind})"))
}

/// A named placeholder in a grammar.
#[pyclass(name = "Extern", module = "glrmask", frozen, from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PyExtern {
    pub(super) name: String,
    pub(super) kind: String,
}

#[pymethods]
impl PyExtern {
    /// The extern's name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// What the extern can be bound to: `"token"` or `"grammar"`.
    ///
    /// A token extern takes only a token ID or token set.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// A representation for debugging.
    /// A representation for debugging.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        extern_repr(py, &self.name, &self.kind)
    }

    /// Whether both have the same name and kind.
    /// Whether both list the same externs in the same order.
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

impl From<glrmask::Extern> for PyExtern {
    fn from(e: glrmask::Extern) -> Self {
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

/// A grammar's or module's externs, in declaration order.
///
/// A read-only map from name to `Extern`.
#[pyclass(name = "Externs", module = "glrmask", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub(super) struct PyExterns {
    pub(super) entries: Vec<PyExtern>,
    pub(super) unbound_entries: Option<Vec<PyExtern>>,
}

#[pymethods]
impl PyExterns {
    /// The externs not yet bound.
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

    /// The number of externs.
    fn __len__(&self) -> usize {
        self.entries.len()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.entries.iter().any(|e| e.name == key)
    }

    /// The extern named `key`. Raises `KeyError` if there is none.
    fn __getitem__(&self, key: &str) -> PyResult<PyExtern> {
        self.entries
            .iter()
            .find(|e| e.name == key)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    /// The extern named `key`, or `default` if there is none.
    #[pyo3(signature = (key, default = None, /))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        if let Some(e) = self.entries.iter().find(|e| e.name == key) {
            Ok(e.clone().into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }

    /// The extern names, in declaration order.
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

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut pairs: Vec<String> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let key = PyString::new(py, &entry.name).repr()?.to_string();
            pairs.push(format!("{key}: {}", extern_repr(py, &entry.name, &entry.kind)?));
        }
        Ok(format!("Externs({{{}}})", pairs.join(", ")))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match other.extract::<PyRef<'_, Self>>() {
            Ok(other) => Ok(pyo3::types::PyBool::new(py, self.entries == other.entries)
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
    /// Matches exactly `value`. A `str` is UTF-8 encoded.
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
    ///
    /// Raises `ValueError` if a part has unbound externs.
    #[staticmethod]
    #[pyo3(signature = (*parts))]
    fn choice(parts: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let items = collect_parts(parts)?;
        Ok(Self {
            inner: glrmask::Expr::choice(items),
        })
    }

    /// Matches `parts` in order.
    ///
    /// Raises `ValueError` if a part has unbound externs.
    #[staticmethod]
    #[pyo3(signature = (*parts))]
    fn sequence(parts: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let items = collect_parts(parts)?;
        Ok(Self {
            inner: glrmask::Expr::sequence(items),
        })
    }
}

/// A grammar you can compose with others and partly compile before its externs are bound. Can be saved and loaded.
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

fn extract_token_id(value: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
    if !value.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(None);
    }
    match value.extract::<u32>() {
        Ok(id) => Ok(Some(id)),
        // An integer outside the u32 token-ID domain is a bad value, not a bad type.
        Err(_) => Err(PyValueError::new_err(format!(
            "token ID {value} is out of range; token IDs go from 0 to {}",
            u32::MAX
        ))),
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
    if let Some(id) = extract_token_id(value)? {
        return Ok(glrmask::Binding::from(id));
    }
    Err(PyTypeError::new_err(
        "Module.bind accepts a Grammar, Module, Expr, TokenSet or token ID",
    ))
}

/// One value passed to `Grammar.bind`, before it is applied to the grammar.
pub(super) enum GrammarBinding {
    Grammar(glrmask::Grammar),
    Expr(glrmask::Expr),
    TokenSet(glrmask::TokenSet),
    Token(u32),
}

impl GrammarBinding {
    pub(super) fn bind(
        self,
        grammar: &mut glrmask::Grammar,
        name: &str,
    ) -> glrmask::Result<()> {
        match self {
            Self::Grammar(value) => grammar.bind(name, value),
            Self::Expr(value) => grammar.bind(name, value),
            Self::TokenSet(value) => grammar.bind(name, value),
            Self::Token(value) => grammar.bind(name, value),
        }
    }
}

pub(super) fn extract_grammar_binding(value: &Bound<'_, PyAny>) -> PyResult<GrammarBinding> {
    if let Ok(g) = value.extract::<PyRef<'_, PyGrammar>>() {
        return Ok(GrammarBinding::Grammar(g.inner.clone()));
    }
    if let Ok(e) = value.extract::<PyRef<'_, PyExpr>>() {
        return Ok(GrammarBinding::Expr(e.inner.clone()));
    }
    if let Ok(t) = value.extract::<PyRef<'_, PyTokenSet>>() {
        return Ok(GrammarBinding::TokenSet(t.inner.clone()));
    }
    if let Some(id) = extract_token_id(value)? {
        return Ok(GrammarBinding::Token(id));
    }
    Err(PyTypeError::new_err(
        "Grammar.bind accepts a Grammar, Expr, TokenSet or token ID",
    ))
}

/// The positional-or-keyword forms shared by `Grammar.bind` and `Module.bind`.
pub(super) enum BindArguments<'py> {
    Single(String, Bound<'py, PyAny>),
    Many(Vec<(String, Bound<'py, PyAny>)>),
}

pub(super) fn bind_arguments<'py>(
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
    method: &str,
) -> PyResult<BindArguments<'py>> {
    let has_args = !args.is_empty();
    let has_kwargs = kwargs.map_or(false, |kw| !kw.is_empty());

    if has_args && has_kwargs {
        return Err(PyTypeError::new_err(format!(
            "cannot mix positional and keyword arguments in {method}",
        )));
    }
    if !has_args && !has_kwargs {
        return Err(PyTypeError::new_err(format!(
            "{method} expects either positional (name, value) or keyword bindings",
        )));
    }

    if has_args {
        if args.len() != 2 {
            return Err(PyTypeError::new_err(format!(
                "{method} positional form takes exactly 2 arguments (name, value), got {}",
                args.len()
            )));
        }
        let name: String = args.get_item(0)?.extract()?;
        return Ok(BindArguments::Single(name, args.get_item(1)?));
    }

    let kwargs = kwargs.unwrap();
    let mut bindings = Vec::with_capacity(kwargs.len());
    for (key, val) in kwargs.iter() {
        let name: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("binding names must be strings"))?;
        bindings.push((name, val));
    }
    Ok(BindArguments::Many(bindings))
}

pub(super) fn compile_options(
    mask_strategy: Option<&Bound<'_, PyAny>>,
    boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
) -> PyResult<glrmask::CompileOptions> {
    let mask_strategy = super::resolve_mask_strategy(mask_strategy)?;
    let boundary_mask_strategy = boundary_mask_strategy
        .as_deref()
        .copied()
        .unwrap_or(PyBoundaryMaskStrategy::AUTO);
    Ok(glrmask::CompileOptions::default()
        .mask_strategy(mask_strategy)
        .boundary_mask_strategy(boundary_mask_strategy.into()))
}

#[pymethods]
impl PyModule {
    /// Creates a module from a grammar.
    #[new]
    #[pyo3(signature = (source, /))]
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let grammar = source
            .extract::<PyRef<'_, PyGrammar>>()
            .map_err(|_| PyTypeError::new_err("Module expects Grammar"))?;
        Ok(Self::from_inner(glrmask::Module::new(&grammar.inner)))
    }

    /// Binds the extern `name` to `value`.
    ///
    /// `bind(a=x, b=y)` binds several at once; if one fails, none is bound. Raises `ValueError` if a name is undeclared or already bound, or a value is the wrong kind.
    #[pyo3(signature = (*args, **kwargs))]
    fn bind(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let request = bind_arguments(args, kwargs, "Module.bind")?;
        let mut bindings = Vec::with_capacity(match &request {
            BindArguments::Single(..) => 1,
            BindArguments::Many(items) => items.len(),
        });
        match request {
            BindArguments::Single(name, value) => {
                bindings.push((name, extract_binding(&value)?));
            }
            BindArguments::Many(items) => {
                for (name, value) in items {
                    bindings.push((name, extract_binding(&value)?));
                }
            }
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

    /// Compiles as much as possible for `vocab` now, even with externs unbound, so later compiles are faster.
    ///
    /// The keyword arguments choose how masks are computed; see `MaskStrategy`, `TokenTrie` and `BoundaryMaskStrategy`.
    #[pyo3(signature = (vocab, *, mask_strategy = None, boundary_mask_strategy = None))]
    fn precompile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<&Bound<'_, PyAny>>,
        boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
    ) -> PyResult<()> {
        let vocab = vocab.inner.clone();
        let options = compile_options(mask_strategy, boundary_mask_strategy)?;
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

    /// Compiles the module for `vocab`, reusing any precompiled work. Every extern must be bound.
    ///
    /// The keyword arguments choose how masks are computed; see `MaskStrategy`, `TokenTrie` and `BoundaryMaskStrategy`.
    #[pyo3(signature = (vocab, *, mask_strategy = None, boundary_mask_strategy = None))]
    fn compile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<&Bound<'_, PyAny>>,
        boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
    ) -> PyResult<PyConstraint> {
        let snapshot = slf.try_borrow()?.snapshot();
        let vocab = vocab.inner.clone();
        let options = compile_options(mask_strategy, boundary_mask_strategy)?;
        let constraint = py
            .detach(move || snapshot.compile_with(&vocab, options))
            .map_err(api_error)?;
        Ok(super::constraint(constraint))
    }

    /// The module's externs.
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

    /// A copy of the module. Binding the copy does not change the original.
    fn copy(&self) -> Self {
        Self::from_inner(self.snapshot())
    }

    /// Same as `copy()`.
    fn __copy__(&self) -> Self {
        self.copy()
    }

    /// Same as `copy()`.
    fn __deepcopy__(slf: &Bound<'_, Self>, memo: &Bound<'_, PyDict>) -> PyResult<Py<Self>> {
        let copy = Py::new(slf.py(), slf.try_borrow()?.copy())?;
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        Ok(copy)
    }

    /// Saves the module as bytes, with its precompiled work but without the vocabulary. Bytes saved by one GLRMask version may not load in another.
    fn save<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyBytes>> {
        let snapshot = slf.try_borrow()?.snapshot();
        let bytes = slf.py().detach(move || snapshot.save());
        Ok(PyBytes::new(slf.py(), &bytes))
    }

    /// Loads a module saved with `save()`. Its precompiled work only applies to the vocabulary it was built for; compiling with another is an error.
    #[staticmethod]
    #[pyo3(signature = (data, /))]
    fn load(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        let data = data.to_vec();
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
