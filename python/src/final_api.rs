//! Python's source/module/root lifecycle, backed by the production Rust API.
mod canonical;
use super::{PyConstraint, PyVocab};
use glrmask::__private::ConstraintExt as _;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyModule};
use std::sync::Arc;

pub(super) fn api_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn constraint(inner: glrmask::Constraint) -> PyConstraint {
    let max_token = inner.max_original_token_id().unwrap_or(0);
    PyConstraint { inner: Arc::new(inner), max_token }
}

/// How masks are computed.
#[allow(non_camel_case_types)]
#[pyclass(name = "MaskStrategy", module = "glrmask", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyMaskStrategy {
    AUTO,
    PARSER_DWA,
    TOKEN_TRIE,
}

impl From<PyMaskStrategy> for glrmask::MaskStrategy {
    fn from(value: PyMaskStrategy) -> Self {
        match value {
            PyMaskStrategy::AUTO => Self::Auto,
            PyMaskStrategy::PARSER_DWA => Self::ParserDwa,
            PyMaskStrategy::TOKEN_TRIE => Self::TokenTrie(glrmask::TokenTrie::new()),
        }
    }
}

impl From<glrmask::MaskStrategy> for PyMaskStrategy {
    fn from(value: glrmask::MaskStrategy) -> Self {
        match value {
            glrmask::MaskStrategy::Auto => Self::AUTO,
            glrmask::MaskStrategy::ParserDwa => Self::PARSER_DWA,
            glrmask::MaskStrategy::TokenTrie(_) => Self::TOKEN_TRIE,
            _ => Self::AUTO,
        }
    }
}

/// How masks are computed at grammar boundaries.
#[allow(non_camel_case_types)]
#[pyclass(name = "BoundaryMaskStrategy", module = "glrmask", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyBoundaryMaskStrategy {
    AUTO,
    PARSER_DWA,
    TOKEN_TRIE,
}

impl From<PyBoundaryMaskStrategy> for glrmask::BoundaryMaskStrategy {
    fn from(value: PyBoundaryMaskStrategy) -> Self {
        match value {
            PyBoundaryMaskStrategy::AUTO => Self::Auto,
            PyBoundaryMaskStrategy::PARSER_DWA => Self::ParserDwa,
            PyBoundaryMaskStrategy::TOKEN_TRIE => Self::TokenTrie,
        }
    }
}

impl From<glrmask::BoundaryMaskStrategy> for PyBoundaryMaskStrategy {
    fn from(value: glrmask::BoundaryMaskStrategy) -> Self {
        match value {
            glrmask::BoundaryMaskStrategy::Auto => Self::AUTO,
            glrmask::BoundaryMaskStrategy::ParserDwa => Self::PARSER_DWA,
            glrmask::BoundaryMaskStrategy::TokenTrie => Self::TOKEN_TRIE,
            _ => Self::AUTO,
        }
    }
}

/// How the parser is represented and run when masks are computed with `MaskStrategy.TOKEN_TRIE`.
#[allow(non_camel_case_types)]
#[pyclass(name = "ParserBackend", module = "glrmask", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum PyParserBackend {
    /// An LR table. Faster to compile, with slower masks. The default.
    LR_TABLE,
    /// Template DFAs. Slower to compile, with faster masks.
    TEMPLATE_DFA,
}

impl From<PyParserBackend> for glrmask::ParserBackend {
    fn from(value: PyParserBackend) -> Self {
        match value {
            PyParserBackend::LR_TABLE => Self::LrTable,
            PyParserBackend::TEMPLATE_DFA => Self::TemplateDfa,
        }
    }
}

impl TryFrom<glrmask::ParserBackend> for PyParserBackend {
    type Error = PyErr;

    fn try_from(value: glrmask::ParserBackend) -> PyResult<Self> {
        match value {
            glrmask::ParserBackend::LrTable => Ok(Self::LR_TABLE),
            glrmask::ParserBackend::TemplateDfa => Ok(Self::TEMPLATE_DFA),
            _ => Err(PyValueError::new_err("parser backend is not supported by this Python binding")),
        }
    }
}

/// Settings for `MaskStrategy.TOKEN_TRIE`. Pass one as `mask_strategy`.
///
/// `TokenTrie()` is the same as `MaskStrategy.TOKEN_TRIE`.
#[pyclass(name = "TokenTrie", module = "glrmask", frozen, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PyTokenTrie {
    inner: glrmask::TokenTrie,
    parser_backend: PyParserBackend,
    merge_equivalent_tokens: bool,
}

#[pymethods]
impl PyTokenTrie {
    #[new]
    #[pyo3(signature = (*, parser_backend = None, merge_equivalent_tokens = false))]
    fn new(
        parser_backend: Option<PyRef<'_, PyParserBackend>>,
        merge_equivalent_tokens: bool,
    ) -> Self {
        let backend = parser_backend
            .as_deref()
            .copied()
            .unwrap_or(PyParserBackend::LR_TABLE);
        Self {
            inner: glrmask::TokenTrie::new()
                .parser_backend(backend.into())
                .merge_equivalent_tokens(merge_equivalent_tokens),
            parser_backend: backend,
            merge_equivalent_tokens,
        }
    }

    /// How the parser is represented. Defaults to `ParserBackend.LR_TABLE`.
    #[getter]
    fn parser_backend(&self) -> PyParserBackend {
        self.parser_backend
    }

    /// Whether tokens that the grammar treats the same are merged, so masks walk fewer tokens. Defaults to `False`.
    ///
    /// Tokens are not merged in a composition that includes precompiled parts.
    #[getter]
    fn merge_equivalent_tokens(&self) -> bool {
        self.merge_equivalent_tokens
    }

    /// Whether both sets contain the same IDs.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match other.extract::<PyRef<'_, Self>>() {
            Ok(other) => Ok(pyo3::types::PyBool::new(py, other.inner == self.inner)
                .to_owned()
                .into_any()
                .unbind()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenTrie(parser_backend=ParserBackend.{:?}, merge_equivalent_tokens={})",
            self.parser_backend,
            if self.merge_equivalent_tokens { "True" } else { "False" }
        )
    }
}

/// Resolve the Python-facing `mask_strategy` union.
pub(super) fn resolve_mask_strategy(value: Option<&Bound<'_, PyAny>>) -> PyResult<glrmask::MaskStrategy> {
    resolve_mask_strategy_settings(value).map(|(strategy, _, _)| strategy)
}

pub(super) fn resolve_mask_strategy_settings(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<(glrmask::MaskStrategy, PyParserBackend, bool)> {
    let Some(value) = value else {
        return Ok((glrmask::MaskStrategy::Auto, PyParserBackend::LR_TABLE, false));
    };
    if let Ok(strategy) = value.extract::<PyRef<'_, PyMaskStrategy>>() {
        return Ok(((*strategy).into(), PyParserBackend::LR_TABLE, false));
    }
    if let Ok(settings) = value.extract::<PyRef<'_, PyTokenTrie>>() {
        return Ok((
            glrmask::MaskStrategy::TokenTrie(settings.inner),
            settings.parser_backend,
            settings.merge_equivalent_tokens,
        ));
    }
    Err(PyTypeError::new_err(
        "mask_strategy must be a MaskStrategy or TokenTrie",
    ))
}

/// A set of token IDs.
#[pyclass(name = "TokenSet", module = "glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyTokenSet {
    pub(super) inner: glrmask::TokenSet,
}

fn python_range(value: &Bound<'_, PyAny>) -> PyResult<Option<std::ops::RangeInclusive<u32>>> {
    let range_type = value.py().import("builtins")?.getattr("range")?;
    if !value.is_instance(&range_type)? {
        return Err(PyTypeError::new_err("TokenSet.from_ranges expects range objects"));
    }
    let step: i128 = value.getattr("step")?.extract()?;
    if step != 1 {
        return Err(PyValueError::new_err("TokenSet.from_ranges requires ranges with step 1"));
    }
    let start: i128 = value.getattr("start")?.extract()?;
    let stop: i128 = value.getattr("stop")?.extract()?;
    if start >= stop {
        return Ok(None);
    }
    let convert = |id: i128| u32::try_from(id).map_err(|_| {
        PyValueError::new_err("TokenSet.from_ranges range contains an ID outside the u32 token-ID domain")
    });
    Ok(Some(convert(start)?..=convert(stop - 1)?))
}

#[pymethods]
impl PyTokenSet {
    /// Creates a set from `ids`.
    #[new]
    #[pyo3(signature = (ids, /))]
    fn new(ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ids = ids
            .try_iter()?
            .map(|item| item?.extract::<u32>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: glrmask::TokenSet::new(ids),
        })
    }

    /// Creates a set from `range` objects, which must have step 1.
    #[staticmethod]
    #[pyo3(signature = (ranges, /))]
    fn from_ranges(ranges: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ranges = ranges
            .try_iter()?
            .map(|item| python_range(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { inner: glrmask::TokenSet::from_ranges(ranges.into_iter().flatten()) })
    }

    /// An expression that rebuilds the set, or a summary for large sets.
    fn __repr__(&self) -> String {
        let ranges = self.inner.range_pairs();
        if ranges.len() > 8 {
            let tokens: u64 = ranges
                .iter()
                .map(|&(start, end)| u64::from(end - start) + 1)
                .sum();
            format!(
                "<glrmask.TokenSet: {tokens} tokens in {} ranges>",
                ranges.len()
            )
        } else {
            let listed: Vec<String> = ranges
                .iter()
                .map(|&(start, end)| format!("range({start}, {})", u64::from(end) + 1))
                .collect();
            format!("TokenSet.from_ranges([{}])", listed.join(", "))
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match other.extract::<PyRef<'_, Self>>() {
            Ok(other) => Ok(pyo3::types::PyBool::new(py, self.inner == other.inner)
                .to_owned()
                .into_any()
                .unbind()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }
}

#[derive(Clone, Copy)]
enum SourceKind { Glrm, JsonSchema, Lark }

/// A grammar, parsed from source or built from an Expr.
#[pyclass(name = "Grammar", module = "glrmask", from_py_object)]
#[derive(Clone)]
pub(super) struct PyGrammar {
    pub(super) inner: glrmask::Grammar,
}

impl PyGrammar {
    fn new(source: String, kind: SourceKind) -> PyResult<Self> {
        let inner = match kind {
            SourceKind::Glrm => glrmask::Grammar::from_glrm(source),
            SourceKind::JsonSchema => glrmask::Grammar::from_json_schema(source),
            SourceKind::Lark => glrmask::Grammar::from_lark(source),
        }
        .map_err(api_error)?;
        Ok(Self { inner })
    }
    pub(super) fn as_rust(&self) -> glrmask::Result<glrmask::Grammar> {
        Ok(self.inner.clone())
    }
    fn replace(&mut self, py: Python<'_>, next: glrmask::Grammar) {
        let previous = std::mem::replace(&mut self.inner, next);
        py.detach(move || drop(previous));
    }
}

impl Drop for PyGrammar {
    fn drop(&mut self) {
        let inner = std::mem::replace(
            &mut self.inner,
            glrmask::Grammar::lark(""),
        );
        let _ = Python::try_attach(move |py| py.detach(move || drop(inner)));
    }
}

#[pymethods]
impl PyGrammar {
    /// Creates a grammar from GLRM source.
    #[staticmethod]
    #[pyo3(signature = (source, /))]
    fn from_glrm(source: String) -> PyResult<Self> {
        Self::new(source, SourceKind::Glrm)
    }

    /// Creates a grammar from a JSON Schema, given as JSON text or a JSON-compatible object such as a dict.
    #[staticmethod]
    #[pyo3(signature = (schema, /))]
    fn from_json_schema(schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = match schema.extract::<String>() {
            Ok(source) => source,
            Err(_) => schema
                .py()
                .import("json")?
                .getattr("dumps")?
                .call1((schema,))?
                .extract()?,
        };
        Self::new(source, SourceKind::JsonSchema)
    }

    /// Creates a grammar from Lark source.
    #[staticmethod]
    #[pyo3(signature = (source, /))]
    fn from_lark(source: String) -> PyResult<Self> {
        Self::new(source, SourceKind::Lark)
    }

    /// Creates a grammar from an expression.
    #[staticmethod]
    #[pyo3(signature = (expr, /))]
    fn from_expr(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let expr = canonical::extract_expr_part(expr)?;
        Ok(Self {
            inner: glrmask::Grammar::from_expr(expr).map_err(api_error)?,
        })
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
        let request = canonical::bind_arguments(args, kwargs, "Grammar.bind")?;
        let mut bindings = Vec::with_capacity(match &request {
            canonical::BindArguments::Single(..) => 1,
            canonical::BindArguments::Many(items) => items.len(),
        });
        match request {
            canonical::BindArguments::Single(name, value) => {
                bindings.push((name, canonical::extract_grammar_binding(&value)?));
            }
            canonical::BindArguments::Many(items) => {
                for (name, value) in items {
                    bindings.push((name, canonical::extract_grammar_binding(&value)?));
                }
            }
        }
        let mut target = slf.try_borrow_mut()?;
        let mut next = target.inner.clone();
        let next = py
            .detach(move || {
                for (name, binding) in bindings {
                    binding.bind(&mut next, &name)?;
                }
                Ok::<_, glrmask::Error>(next)
            })
            .map_err(api_error)?;
        target.replace(py, next);
        Ok(())
    }

    /// The grammar's externs.
    #[getter]
    fn externs(&self) -> PyResult<canonical::PyExterns> {
        let inner = self.inner.clone();
        let all: Vec<canonical::PyExtern> = inner.externs().iter().map(canonical::PyExtern::from).collect();
        let unbound: Vec<canonical::PyExtern> = inner.externs().unbound().iter().map(canonical::PyExtern::from).collect();
        Ok(canonical::PyExterns { entries: all, unbound_entries: Some(unbound) })
    }

    /// Compiles the grammar for `vocab`. Raises `ValueError` if any extern is unbound.
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
        let grammar = slf.try_borrow()?.inner.clone();
        let vocab = vocab.inner.clone();
        let options = canonical::compile_options(mask_strategy, boundary_mask_strategy)?;
        py.detach(move || {
            let module = glrmask::Module::new(&grammar);
            module.compile_with(&vocab, options)
        })
        .map(constraint)
        .map_err(api_error)
    }

    /// Compiles as much as possible for `vocab` now, even with externs unbound, and returns a `Module` that keeps the work.
    ///
    /// The keyword arguments choose how masks are computed; see `MaskStrategy`, `TokenTrie` and `BoundaryMaskStrategy`.
    #[pyo3(signature = (vocab, *, mask_strategy = None, boundary_mask_strategy = None))]
    fn precompile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<&Bound<'_, PyAny>>,
        boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
    ) -> PyResult<canonical::PyModule> {
        let grammar = slf.try_borrow()?.inner.clone();
        let vocab = vocab.inner.clone();
        let options = canonical::compile_options(mask_strategy, boundary_mask_strategy)?;
        let module = py
            .detach(move || {
                let mut module = glrmask::Module::new(&grammar);
                module.precompile_with(&vocab, options)?;
                Ok::<_, glrmask::Error>(module)
            })
            .map_err(api_error)?;
        Ok(canonical::PyModule::from_inner(module))
    }

    /// A copy of the grammar. Binding the copy does not change the original.
    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
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
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    canonical::register(module)?;
    module.add_class::<PyGrammar>()?;
    module.add_class::<PyTokenSet>()?;
    module.add_class::<PyMaskStrategy>()?;
    module.add_class::<PyBoundaryMaskStrategy>()?;
    module.add_class::<PyParserBackend>()?;
    module.add_class::<PyTokenTrie>()?;

    let mask_strat = module.getattr("MaskStrategy")?;
    mask_strat.setattr("FAST_COMPILE", mask_strat.getattr("TOKEN_TRIE")?)?;
    mask_strat.setattr("FAST_MASKS", mask_strat.getattr("PARSER_DWA")?)?;

    let boundary_strat = module.getattr("BoundaryMaskStrategy")?;
    boundary_strat.setattr("FAST_COMPILE", boundary_strat.getattr("TOKEN_TRIE")?)?;
    boundary_strat.setattr("FAST_MASKS", boundary_strat.getattr("PARSER_DWA")?)?;

    Ok(())
}

/// Tooling-only inspection of the stored parser representation.
#[pyfunction]
fn parser_backend(constraint: &PyConstraint) -> PyResult<PyParserBackend> {
    constraint.inner.parser_backend().try_into()
}

#[pyfunction]
#[pyo3(signature = (grammar, vocab, *, end_tokens = None, mask_strategy = None))]
fn compile_with_backend(
    grammar: &PyGrammar,
    vocab: &PyVocab,
    py: Python<'_>,
    end_tokens: Option<Vec<u32>>,
    mask_strategy: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyConstraint> {
    let (strategy, backend, merge_equivalent_tokens) =
        resolve_mask_strategy_settings(mask_strategy)?;
    let options = glrmask::BuildOptions::default()
        .end_tokens(end_tokens.unwrap_or_default())
        .optimization(strategy)
        .parser_backend(glrmask::ParserBackend::TemplateDfa)
        .engine_backend(backend.into())
        .merge_equivalent_tokens(merge_equivalent_tokens);
    let grammar = grammar.as_rust().map_err(api_error)?;
    py.detach(|| grammar.compile_with_build_options(&vocab.inner, options))
        .map(constraint)
        .map_err(api_error)
}

pub(super) fn register_internal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParserBackend>()?;
    module.add_function(wrap_pyfunction!(parser_backend, module)?)?;
    module.add_function(wrap_pyfunction!(compile_with_backend, module)?)?;
    super::template_api::register(module)?;
    Ok(())
}
