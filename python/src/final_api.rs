//! Python's source/module/root lifecycle, backed by the production Rust API.
use super::{PyConstraint, PyVocab};
use glrmask::__private::ConstraintExt as _;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule};
use std::collections::BTreeMap;
use std::sync::Arc;

pub(super) fn api_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn constraint(inner: glrmask::Constraint) -> PyConstraint {
    let max_token = inner.max_original_token_id().unwrap_or(0);
    PyConstraint { inner: Arc::new(inner), max_token }
}

/// Preferred compilation/runtime trade-off; accepted-language semantics are unchanged.
#[allow(non_camel_case_types)]
#[pyclass(name = "Optimization", module = "glrmask", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyOptimization {
    AUTO,
    FAST_BUILD,
    FAST_RUNTIME,
}

/// Optional boundary-query metadata, built before compile or link returns.
#[allow(non_camel_case_types)]
#[pyclass(name = "BoundaryTriggerDetail", module = "glrmask", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyBoundaryTriggerDetail { NONE, TOKENS, EXACT }

impl From<PyBoundaryTriggerDetail> for glrmask::BoundaryTriggerDetail {
    fn from(value: PyBoundaryTriggerDetail) -> Self {
        match value { PyBoundaryTriggerDetail::NONE => Self::None,
            PyBoundaryTriggerDetail::TOKENS => Self::Tokens,
            PyBoundaryTriggerDetail::EXACT => Self::Exact }
    }
}

fn with_boundary(options: glrmask::BuildOptions, detail: Option<PyRef<'_, PyBoundaryTriggerDetail>>) -> glrmask::BuildOptions {
    match detail { Some(detail) => options.boundary_trigger((*detail).into()), None => options }
}

/// Parser execution and storage backend, independent of mask optimization.
#[allow(non_camel_case_types)]
#[pyclass(name = "ParserBackend", module = "glrmask._internal", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyParserBackend {
    LR_TABLE,
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

pub(super) fn optimization(value: PyOptimization) -> glrmask::Optimization {
    match value {
        PyOptimization::AUTO => glrmask::Optimization::Auto,
        PyOptimization::FAST_BUILD => glrmask::Optimization::FastBuild,
        PyOptimization::FAST_RUNTIME => glrmask::Optimization::FastRuntime,
    }
}

fn options(
    end_tokens: Option<Vec<u32>>,
    optimization: Option<PyRef<'_, PyOptimization>>,
    parser_backend: Option<PyRef<'_, PyParserBackend>>,
) -> glrmask::BuildOptions {
    let mode = match optimization.as_deref().copied().unwrap_or(PyOptimization::AUTO) {
        PyOptimization::AUTO => glrmask::Optimization::Auto,
        PyOptimization::FAST_BUILD => glrmask::Optimization::FastBuild,
        PyOptimization::FAST_RUNTIME => glrmask::Optimization::FastRuntime,
    };
    let backend = parser_backend.as_deref().copied().unwrap_or(PyParserBackend::TEMPLATE_DFA);
    glrmask::BuildOptions::default().end_tokens(end_tokens.unwrap_or_default())
        .optimization(mode).parser_backend(backend.into())
}

/// One exact model token, retaining its complete vocabulary identity.
#[pyclass(name = "ExactToken", module = "glrmask", frozen)]
#[derive(Clone)]
pub(super) struct PyExactToken {
    pub(super) inner: glrmask::ExactToken,
}

#[pymethods]
impl PyExactToken {
    #[getter]
    fn id(&self) -> u32 { self.inner.id() }
}

/// A nonempty set of exact model tokens, retaining its vocabulary identity.
#[pyclass(name = "ExactTokens", module = "glrmask", frozen)]
#[derive(Clone)]
pub(super) struct PyExactTokens {
    pub(super) inner: glrmask::ExactTokens,
}

#[pymethods]
impl PyExactTokens {
    #[getter]
    fn ids(&self) -> Vec<u32> { self.inner.ids().to_vec() }
}

#[derive(Clone, Copy)]
enum SourceKind { Ebnf, Lark, Glrm, JsonSchema }

#[derive(Clone)]
enum Binding {
    Source(Box<PyGrammar>),
    Constraint(Arc<glrmask::Constraint>),
    Token(glrmask::ExactToken),
    Tokens(glrmask::ExactTokens),
}

impl Binding {
    fn extract_for_grammar(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = value.extract::<PyRef<'_, PyGrammar>>() {
            return Ok(Self::Source(Box::new(value.clone())));
        }
        if let Ok(value) = value.extract::<PyRef<'_, PyExactToken>>() {
            return Ok(Self::Token(value.inner.clone()));
        }
        if let Ok(value) = value.extract::<PyRef<'_, PyExactTokens>>() {
            return Ok(Self::Tokens(value.inner.clone()));
        }
        Err(PyTypeError::new_err("Grammar.bind accepts Grammar, ExactToken, or ExactTokens"))
    }

    fn extract_for_unlinked(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = value.extract::<PyRef<'_, PyConstraint>>() {
            return Ok(Self::Constraint(Arc::clone(&value.inner)));
        }
        if let Ok(value) = value.extract::<PyRef<'_, PyExactToken>>() {
            return Ok(Self::Token(value.inner.clone()));
        }
        if let Ok(value) = value.extract::<PyRef<'_, PyExactTokens>>() {
            return Ok(Self::Tokens(value.inner.clone()));
        }
        Err(PyTypeError::new_err(
            "UnlinkedConstraint.bind accepts Constraint, ExactToken, or ExactTokens",
        ))
    }

    fn apply<'a>(&'a self, grammar: glrmask::Grammar<'a>, name: &str) -> glrmask::Result<glrmask::Grammar<'a>> {
        match self {
            Self::Source(child) => grammar.bind(name, child.as_rust()?),
            Self::Token(token) => grammar.bind(name, token.clone()),
            Self::Tokens(tokens) => grammar.bind(name, tokens.clone()),
            Self::Constraint(_) => unreachable!("compiled bindings are rejected by Grammar.bind"),
        }
    }
}

/// Immutable grammar description with source-grammar or exact-token bindings.
///
/// bind returns another description. compile produces a complete runnable
/// Constraint; compile_unlinked intentionally preserves unresolved slots.
#[pyclass(name = "Grammar", module = "glrmask", frozen)]
#[derive(Clone)]
pub(super) struct PyGrammar {
    source: String,
    kind: SourceKind,
    bindings: BTreeMap<String, Binding>,
}

impl PyGrammar {
    fn new(source: String, kind: SourceKind) -> Self {
        Self { source, kind, bindings: BTreeMap::new() }
    }

    fn as_rust(&self) -> glrmask::Result<glrmask::Grammar<'_>> {
        let mut grammar = match self.kind {
            SourceKind::Ebnf => glrmask::Grammar::from_ebnf(&self.source),
            SourceKind::Lark => glrmask::Grammar::from_lark(&self.source),
            SourceKind::Glrm => glrmask::Grammar::from_glrm(&self.source),
            SourceKind::JsonSchema => glrmask::Grammar::from_json_schema(&self.source),
        };
        for (name, value) in &self.bindings { grammar = value.apply(grammar, name)?; }
        Ok(grammar)
    }
}

#[pymethods]
impl PyGrammar {
    /// Create a grammar from EBNF source text.
    #[staticmethod]
    fn from_ebnf(source: String) -> Self { Self::new(source, SourceKind::Ebnf) }

    /// Create a grammar from Lark source text.
    #[staticmethod]
    fn from_lark(source: String) -> Self { Self::new(source, SourceKind::Lark) }

    /// Create a grammar from native GLRM source text.
    #[staticmethod]
    fn from_glrm(source: String) -> Self { Self::new(source, SourceKind::Glrm) }

    /// Create a grammar from JSON Schema text or a JSON-serializable object.
    #[staticmethod]
    fn from_json_schema(schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = match schema.extract::<String>() {
            Ok(source) => source,
            Err(_) => schema.py().import("json")?.getattr("dumps")?.call1((schema,))?.extract()?,
        };
        Ok(Self::new(source, SourceKind::JsonSchema))
    }

    /// Return a new description with one declared slot bound. No compilation occurs.
    fn bind(&self, name: String, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if self.bindings.contains_key(&name) {
            return Err(PyValueError::new_err(format!("external {name:?} was bound more than once")));
        }
        let mut next = self.clone();
        next.bindings.insert(name, Binding::extract_for_grammar(value)?);
        next.as_rust().map_err(api_error)?; // Validate declaration/kind before storing the description.
        Ok(next)
    }

    /// Compile a closed, immediately runnable constraint for this vocabulary.
    ///
    /// end_tokens are reserved generation controls, allowed only at acceptance.
    /// Their policy is not inherited when the result is later embedded as a child.
    #[pyo3(signature = (vocab, *, end_tokens=None, optimization=None, boundary_trigger=None))]
    fn compile(
        &self, py: Python<'_>, vocab: &PyVocab,
        end_tokens: Option<Vec<u32>>, optimization: Option<PyRef<'_, PyOptimization>>,
        boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>,
    ) -> PyResult<PyConstraint> {
        let options = with_boundary(options(end_tokens, optimization, None), boundary_trigger);
        let grammar = self.as_rust().map_err(api_error)?;
        let vocab = vocab.inner.clone();
        py.allow_threads(move || grammar.compile_with(&vocab, options))
            .map(constraint).map_err(api_error)
    }

    /// Compile reusable machinery while preserving unresolved grammar/token slots.
    #[pyo3(signature = (vocab, *, boundary_trigger=None))]
    fn compile_unlinked(&self, py: Python<'_>, vocab: &PyVocab,
        boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>) -> PyResult<PyUnlinkedConstraint> {
        let options = with_boundary(glrmask::BuildOptions::default(), boundary_trigger);
        let grammar = self.as_rust().map_err(api_error)?;
        let vocab = vocab.inner.clone();
        py.allow_threads(move || grammar.compile_unlinked_with(&vocab, options))
            .map(|inner| PyUnlinkedConstraint { inner }).map_err(api_error)
    }
}

/// Immutable vocabulary-specific compiled constraint in pre-link form.
#[pyclass(name = "UnlinkedConstraint", module = "glrmask", frozen)]
#[derive(Clone)]
pub(super) struct PyUnlinkedConstraint {
    inner: glrmask::UnlinkedConstraint,
}

#[pymethods]
impl PyUnlinkedConstraint {
    /// Bind a compiled Constraint child or exact-token value, returning a new value.
    fn bind(&self, py: Python<'_>, name: String, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let binding = Binding::extract_for_unlinked(value)?;
        let module = self.inner.clone();
        let result = py.allow_threads(move || match binding {
            Binding::Constraint(child) => module.bind(&name, child),
            Binding::Token(token) => module.bind(&name, token),
            Binding::Tokens(tokens) => module.bind(&name, tokens),
            Binding::Source(_) => unreachable!("source binding rejected above"),
        });
        result.map(|inner| Self { inner }).map_err(api_error)
    }

    /// Produce a runnable root. Every required slot must already be bound.
    #[pyo3(signature = (*, end_tokens=None, optimization=None, boundary_trigger=None))]
    fn link(
        &self, py: Python<'_>, end_tokens: Option<Vec<u32>>,
        optimization: Option<PyRef<'_, PyOptimization>>,
        boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>,
    ) -> PyResult<PyConstraint> {
        let options = with_boundary(options(end_tokens, optimization, None), boundary_trigger);
        py.allow_threads(|| self.inner.link_with(options)).map(constraint).map_err(api_error)
    }

    /// Serialize compiled machinery and its open-slot manifest as bytes.
    fn save<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = py.allow_threads(|| self.inner.save());
        PyBytes::new(py, &bytes)
    }

    /// Load an unlinked constraint, retaining open slots and exact vocabulary identity.
    #[staticmethod]
    #[pyo3(signature = (data, vocab=None))]
    fn load(py: Python<'_>, data: &[u8], vocab: Option<&PyVocab>) -> PyResult<Self> {
        let data = data.to_vec();
        let vocab = vocab.map(|v| v.inner.clone());
        py.allow_threads(move || match vocab {
            Some(vocab) => glrmask::UnlinkedConstraint::load_with_vocab(data, &vocab),
            None => glrmask::UnlinkedConstraint::load(data),
        }).map(|inner| Self { inner }).map_err(api_error)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGrammar>()?;
    module.add_class::<PyUnlinkedConstraint>()?;
    module.add_class::<PyExactToken>()?;
    module.add_class::<PyExactTokens>()?;
    module.add_class::<PyOptimization>()?;
    module.add_class::<PyBoundaryTriggerDetail>()?;
    Ok(())
}

/// Tooling-only inspection of the stored parser representation.
#[pyfunction]
fn parser_backend(constraint: &PyConstraint) -> PyResult<PyParserBackend> {
    constraint.inner.parser_backend().try_into()
}

#[pyfunction]
#[pyo3(signature = (grammar, vocab, *, end_tokens=None, optimization=None, parser_backend=None))]
fn compile_with_backend(
    grammar: &PyGrammar, vocab: &PyVocab, py: Python<'_>,
    end_tokens: Option<Vec<u32>>, optimization: Option<PyRef<'_, PyOptimization>>,
    parser_backend: Option<PyRef<'_, PyParserBackend>>,
) -> PyResult<PyConstraint> {
    let options = options(end_tokens, optimization, parser_backend);
    let grammar = grammar.as_rust().map_err(api_error)?;
    py.allow_threads(|| grammar.compile_with(&vocab.inner, options)).map(constraint).map_err(api_error)
}

#[pyfunction]
#[pyo3(signature = (module, *, end_tokens=None, optimization=None, parser_backend=None))]
fn link_with_backend(
    module: &PyUnlinkedConstraint, py: Python<'_>, end_tokens: Option<Vec<u32>>,
    optimization: Option<PyRef<'_, PyOptimization>>,
    parser_backend: Option<PyRef<'_, PyParserBackend>>,
) -> PyResult<PyConstraint> {
    let options = options(end_tokens, optimization, parser_backend);
    py.allow_threads(|| module.inner.link_with(options)).map(constraint).map_err(api_error)
}

pub(super) fn register_internal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyParserBackend>()?;
    module.add_function(wrap_pyfunction!(parser_backend, module)?)?;
    module.add_function(wrap_pyfunction!(compile_with_backend, module)?)?;
    module.add_function(wrap_pyfunction!(link_with_backend, module)?)?;
    super::template_api::register(module)?;
    Ok(())
}
