//! Python's source/module/root lifecycle, backed by the production Rust API.
mod canonical;
use super::{PyConstraint, PyVocab};
use glrmask::__private::ConstraintExt as _;
use pyo3::exceptions::PyValueError;
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

/// Boundary mask detail level: minimum metadata to build for composition boundaries.
#[allow(non_camel_case_types)]
#[pyclass(name = "BoundaryMaskStrategy", module = "glrmask", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyBoundaryMaskStrategy {
    FAST_BUILD,
    FAST_RUNTIME,
}

impl From<PyBoundaryMaskStrategy> for glrmask::BoundaryMaskStrategy {
    fn from(value: PyBoundaryMaskStrategy) -> Self {
        match value {
            PyBoundaryMaskStrategy::FAST_BUILD => Self::Dynamic,
            PyBoundaryMaskStrategy::FAST_RUNTIME => Self::ParserDwa,
        }
    }
}

impl From<glrmask::BoundaryMaskStrategy> for PyBoundaryMaskStrategy {
    fn from(value: glrmask::BoundaryMaskStrategy) -> Self {
        match value {
            glrmask::BoundaryMaskStrategy::Dynamic => Self::FAST_BUILD,
            glrmask::BoundaryMaskStrategy::ParserDwa => Self::FAST_RUNTIME,
            _ => Self::FAST_BUILD,
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

/// An original token ID with no vocabulary identity. Only ID membership is checked against the target vocabulary; the same ID can refer to different bytes. Use `vocab.tokens([id])` for a vocabulary-bound singleton.
#[pyclass(name = "Token", module = "glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyToken {
    pub(super) inner: glrmask::Token,
}

#[pymethods]
impl PyToken {
    #[new]
    #[pyo3(signature = (id))]
    fn new(id: u32) -> Self { Self { inner: glrmask::Token::new(id) } }

    #[getter]
    fn id(&self) -> u32 { self.inner.id() }
}

/// One token event chosen from a vocabulary-bound set. Vocabularies must have the same ID-to-byte mapping and exact-only IDs, even for a singleton. An empty set matches nothing.
#[pyclass(name = "ExactTokens", module = "glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyExactTokens {
    pub(super) inner: glrmask::ExactTokens,
}

#[pymethods]
impl PyExactTokens {
    /// Sorted IDs, materialized on access. Prefer `ranges` for large sets.
    #[getter]
    fn ids(&self) -> Vec<u32> { self.inner.ids().to_vec() }

    /// Normalized, disjoint Python ranges with exclusive stops; does not expand the IDs.
    #[getter]
    fn ranges(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let range_type = py.import("builtins")?.getattr("range")?;
        self.inner.ranges().map(|range| {
            range_type.call1((*range.start(), u64::from(*range.end()) + 1)).map(Bound::unbind)
        }).collect()
    }

    /// Number of IDs, rather than ranges.
    fn __len__(&self) -> usize { self.inner.len() }
}

#[derive(Clone, Copy)]
enum SourceKind { Ebnf, Glrm, JsonSchema }

/// Reusable source grammar without compilation settings or bindings.
#[pyclass(name = "Grammar", module = "glrmask", from_py_object)]
#[derive(Clone)]
pub(super) struct PyGrammar {
    pub(super) inner: glrmask::Grammar<'static>,
}

impl PyGrammar {
    fn new(source: String, kind: SourceKind) -> Self {
        let inner = match kind {
            SourceKind::Ebnf => glrmask::Grammar::from_ebnf_owned(source),
            SourceKind::Glrm => glrmask::Grammar::from_glrm_owned(source),
            SourceKind::JsonSchema => glrmask::Grammar::from_json_schema_owned(source),
        };
        Self { inner }
    }
    pub(super) fn as_rust(&self) -> glrmask::Result<glrmask::Grammar<'static>> {
        Ok(self.inner.clone())
    }
}

impl Drop for PyGrammar {
    fn drop(&mut self) {
        let inner = std::mem::replace(
            &mut self.inner,
            glrmask::Grammar::from_ebnf_owned(String::new()),
        );
        let _ = Python::try_attach(move |py| py.detach(move || drop(inner)));
    }
}

#[pymethods]
impl PyGrammar {
    /// Parsing and compilation errors can be deferred until `compile`.
    #[staticmethod]
    fn from_ebnf(source: String) -> Self {
        Self::new(source, SourceKind::Ebnf)
    }

    /// GLRM supports typed external declarations; bind them via `Module(g).bind(...)` before compiling.
    #[staticmethod]
    fn from_glrm(source: String) -> Self {
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
        Ok(Self::new(source, SourceKind::JsonSchema))
    }

    /// The default `recognition` profile and the `llm` extension profile differ in accepted syntax. Named token atoms require explicit `token_names` mappings.
    #[staticmethod]
    #[pyo3(signature = (source, *, profile = "recognition", token_names = None))]
    fn from_lark(
        source: String,
        profile: &str,
        token_names: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut options = match profile {
            "recognition" => glrmask::LarkOptions::default(),
            "llm" => glrmask::LarkOptions::llm(),
            _ => return Err(PyValueError::new_err("Lark profile must be 'recognition' or 'llm'")),
        };
        if let Some(names) = token_names {
            for (name, value) in names.iter() {
                let name: String = name.extract()?;
                let ids = match value.extract::<u32>() {
                    Ok(id) => vec![id],
                    Err(_) => value.extract::<Vec<u32>>()?,
                };
                options = options.token(name, ids);
            }
        }
        Ok(Self {
            inner: glrmask::Grammar::from_lark_owned(source, options),
        })
    }

    /// Compile this grammar against the target vocabulary. Sugar for `Module(self, mask_strategy=..., boundary_mask_strategy=...).compile(vocab, end_tokens=...)`.
    #[pyo3(signature = (vocab, *, end_tokens = None, mask_strategy = None, boundary_mask_strategy = None))]
    fn compile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        end_tokens: Option<Vec<u32>>,
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
            .unwrap_or(PyBoundaryMaskStrategy::FAST_RUNTIME)
            .into();
        let options = glrmask::CompileOptions::default()
            .end_tokens(end_tokens.unwrap_or_default());
        py.detach(move || {
            let mut module = glrmask::Module::new(&grammar);
            module.set_mask_strategy(mask_strategy)?;
            module.set_boundary_mask_strategy(boundary_mask_strategy)?;
            module.compile_with(&vocab, options)
        })
        .map(constraint)
        .map_err(api_error)
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
    module.add_class::<PyToken>()?;
    module.add_class::<PyExactTokens>()?;
    module.add_class::<PyMaskStrategy>()?;
    module.add_class::<PyBoundaryMaskStrategy>()?;

    let mask_strat = module.getattr("MaskStrategy")?;
    mask_strat.setattr("FAST_BUILD", mask_strat.getattr("TOKEN_TRIE")?)?;
    mask_strat.setattr("FAST_RUNTIME", mask_strat.getattr("PARSER_DWA")?)?;

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
