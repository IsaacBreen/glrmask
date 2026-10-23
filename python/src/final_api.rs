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

/// Mask generation strategy: trade off compilation time against per-token masking latency.
#[allow(non_camel_case_types)]
#[pyclass(name = "MaskStrategy", module = "glrmask", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyMaskStrategy {
    AUTO,
    PARSER_DWA,
    TOKEN_TRIE,
    TOKEN_CLASS_TRIE,
}

impl From<PyMaskStrategy> for glrmask::MaskStrategy {
    fn from(value: PyMaskStrategy) -> Self {
        match value {
            PyMaskStrategy::AUTO => Self::Auto,
            PyMaskStrategy::PARSER_DWA => Self::ParserDwa,
            PyMaskStrategy::TOKEN_TRIE => Self::TokenTrie,
            PyMaskStrategy::TOKEN_CLASS_TRIE => Self::TokenClassTrie,
        }
    }
}

impl From<glrmask::MaskStrategy> for PyMaskStrategy {
    fn from(value: glrmask::MaskStrategy) -> Self {
        match value {
            glrmask::MaskStrategy::Auto => Self::AUTO,
            glrmask::MaskStrategy::ParserDwa => Self::PARSER_DWA,
            glrmask::MaskStrategy::TokenTrie => Self::TOKEN_TRIE,
            glrmask::MaskStrategy::TokenClassTrie => Self::TOKEN_CLASS_TRIE,
            _ => Self::AUTO,
        }
    }
}

/// Boundary parser selection for crossings between composed grammars.
///
/// `PARSER_DWA` installs the static boundary parser (faster crossings; an
/// over-budget build raises `ValueError`). `TOKEN_TRIE` keeps the segmented
/// dynamic boundary and builds faster. `AUTO` uses `PARSER_DWA` and falls back
/// to `TOKEN_TRIE` when the static boundary parser declines on budget.
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

/// Parser execution and storage backend, independent of mask optimization.
#[allow(non_camel_case_types)]
#[pyclass(name = "ParserBackend", module = "glrmask._internal", eq, eq_int, from_py_object)]
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

/// A set of model token IDs. IDs are validated against the compile vocabulary; an empty set matches nothing.
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
    #[new]
    #[pyo3(signature = (ids))]
    fn new(ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ids = ids
            .try_iter()?
            .map(|item| item?.extract::<u32>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: glrmask::TokenSet::new(ids),
        })
    }

    /// Creates a set from Python ranges with step 1 and exclusive stops; overlapping or adjacent ranges merge, and an empty or reversed range contributes nothing.
    #[staticmethod]
    fn from_ranges(ranges: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ranges = ranges
            .try_iter()?
            .map(|item| python_range(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { inner: glrmask::TokenSet::from_ranges(ranges.into_iter().flatten()) })
    }

    /// The normalized ranges, for example `TokenSet[0..=9, 20..=29]`.
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[derive(Clone, Copy)]
enum SourceKind { Ebnf, Glrm, JsonSchema, Lark }

/// Reusable source grammar without compilation settings or bindings.
#[pyclass(name = "Grammar", module = "glrmask", from_py_object)]
#[derive(Clone)]
pub(super) struct PyGrammar {
    pub(super) inner: glrmask::Grammar,
}

impl PyGrammar {
    fn new(source: String, kind: SourceKind) -> PyResult<Self> {
        let inner = match kind {
            SourceKind::Ebnf => glrmask::Grammar::from_ebnf(source),
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
}

impl Drop for PyGrammar {
    fn drop(&mut self) {
        let inner = std::mem::replace(
            &mut self.inner,
            glrmask::Grammar::ebnf(""),
        );
        let _ = Python::try_attach(move |py| py.detach(move || drop(inner)));
    }
}

#[pymethods]
impl PyGrammar {
    #[staticmethod]
    fn from_ebnf(source: String) -> PyResult<Self> {
        Self::new(source, SourceKind::Ebnf)
    }

    /// GLRM supports typed external declarations; bind them via `Module(g).bind(...)` before compiling.
    #[staticmethod]
    fn from_glrm(source: String) -> PyResult<Self> {
        Self::new(source, SourceKind::Glrm)
    }

    /// Accepts JSON text or a JSON-serializable object. Unsupported keywords are rejected with an error, except that
    /// `format` is treated as an annotation only (as in JSON Schema 2020-12) and
    /// unrecognised keywords are ignored.
    #[staticmethod]
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

    /// Named token atoms declare token externs; bind each through `Module.bind`.
    #[staticmethod]
    fn from_lark(source: String) -> PyResult<Self> {
        Self::new(source, SourceKind::Lark)
    }

    /// Builds a source grammar from an `Expr`, `Grammar`, or `TokenSet`.
    #[staticmethod]
    fn from_expr(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let expr = canonical::extract_expr_part(value)?;
        Ok(Self {
            inner: glrmask::Grammar::from_expr(expr).map_err(api_error)?,
        })
    }

    /// Binds one declared extern. Accepts `Grammar`, `Expr`, or `TokenSet`.
    fn bind(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        canonical::bind_grammar(&mut self.inner, name, value)
    }

    /// The grammar's declared externs, in declaration order.
    #[getter]
    fn externs(&self) -> PyResult<canonical::PyExterns> {
        let inner = self.inner.clone();
        let all: Vec<canonical::PyExtern> = inner.externs().iter().map(canonical::PyExtern::from).collect();
        let unbound: Vec<canonical::PyExtern> = inner.externs().unbound().iter().map(canonical::PyExtern::from).collect();
        Ok(canonical::PyExterns { entries: all, unbound_entries: Some(unbound) })
    }

    /// Compile this grammar against the target vocabulary. Sugar for `Module(self, mask_strategy=..., boundary_mask_strategy=...).compile(vocab)`.
    #[pyo3(signature = (vocab, *, mask_strategy = None, boundary_mask_strategy = None))]
    fn compile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<PyRef<'_, PyMaskStrategy>>,
        boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
    ) -> PyResult<PyConstraint> {
        let grammar = slf.try_borrow()?.inner.clone();
        let vocab = vocab.inner.clone();
        let mask_strategy: glrmask::MaskStrategy = mask_strategy
            .as_deref()
            .copied()
            .unwrap_or(PyMaskStrategy::AUTO)
            .into();
        let boundary_mask_strategy: glrmask::BoundaryMaskStrategy = boundary_mask_strategy
            .as_deref()
            .copied()
            .unwrap_or(PyBoundaryMaskStrategy::AUTO)
            .into();
        let options = glrmask::CompileOptions::default()
            .mask_strategy(mask_strategy)
            .boundary_mask_strategy(boundary_mask_strategy);
        py.detach(move || {
            let module = glrmask::Module::new(&grammar);
            module.compile_with(&vocab, options)
        })
        .map(constraint)
        .map_err(api_error)
    }

    /// Retains vocabulary-specific compilation artifacts and returns the module.
    ///
    /// Equivalent to `Module(self).precompile(vocab, ...)`.
    #[pyo3(signature = (vocab, *, mask_strategy = None, boundary_mask_strategy = None))]
    fn precompile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<PyRef<'_, PyMaskStrategy>>,
        boundary_mask_strategy: Option<PyRef<'_, PyBoundaryMaskStrategy>>,
    ) -> PyResult<canonical::PyModule> {
        let grammar = slf.try_borrow()?.inner.clone();
        let vocab = vocab.inner.clone();
        let options = canonical::compile_options(mask_strategy, boundary_mask_strategy);
        let module = py
            .detach(move || {
                let mut module = glrmask::Module::new(&grammar);
                module.precompile_with(&vocab, options)?;
                Ok::<_, glrmask::Error>(module)
            })
            .map_err(api_error)?;
        Ok(canonical::PyModule::from_inner(module))
    }

    fn __copy__(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __deepcopy__(slf: &Bound<'_, Self>, memo: &Bound<'_, PyDict>) -> PyResult<Py<Self>> {
        let copy = Py::new(
            slf.py(),
            Self {
                inner: slf.try_borrow()?.inner.clone(),
            },
        )?;
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

    let mask_strat = module.getattr("MaskStrategy")?;
    mask_strat.setattr("FAST_BUILD", mask_strat.getattr("TOKEN_TRIE")?)?;
    mask_strat.setattr("FAST_RUNTIME", mask_strat.getattr("PARSER_DWA")?)?;

    let boundary_strat = module.getattr("BoundaryMaskStrategy")?;
    boundary_strat.setattr("FAST_BUILD", boundary_strat.getattr("TOKEN_TRIE")?)?;
    boundary_strat.setattr("FAST_RUNTIME", boundary_strat.getattr("PARSER_DWA")?)?;

    Ok(())
}

/// Tooling-only inspection of the stored parser representation.
#[pyfunction]
fn parser_backend(constraint: &PyConstraint) -> PyResult<PyParserBackend> {
    constraint.inner.parser_backend().try_into()
}

#[pyfunction]
#[pyo3(signature = (grammar, vocab, *, end_tokens = None, mask_strategy = None, parser_backend = None))]
fn compile_with_backend(
    grammar: &PyGrammar,
    vocab: &PyVocab,
    py: Python<'_>,
    end_tokens: Option<Vec<u32>>,
    mask_strategy: Option<PyRef<'_, PyMaskStrategy>>,
    parser_backend: Option<PyRef<'_, PyParserBackend>>,
) -> PyResult<PyConstraint> {
    let strategy = mask_strategy
        .as_deref()
        .copied()
        .unwrap_or(PyMaskStrategy::AUTO);
    let backend = parser_backend
        .as_deref()
        .copied()
        .unwrap_or(PyParserBackend::TEMPLATE_DFA);
    let options = glrmask::BuildOptions::default()
        .end_tokens(end_tokens.unwrap_or_default())
        .optimization(strategy.into())
        .parser_backend(backend.into());
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
