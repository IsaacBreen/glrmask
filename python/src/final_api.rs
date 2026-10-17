//! Python's source/module/root lifecycle, backed by the production Rust API.
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

/// Build-time versus per-token cost. The modes preserve the supported grammar language; `AUTO` chooses a strategy.
#[allow(non_camel_case_types)]
#[pyclass(name = "Optimization", module = "glrmask", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PyOptimization {
    AUTO,
    FAST_BUILD,
    BALANCED,
    FAST_RUNTIME,
}

/// Optional metadata for composition boundaries, independent of the masking optimization. A request at final link overrides the prepared request.
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
        PyOptimization::BALANCED => glrmask::Optimization::Balanced,
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
        PyOptimization::BALANCED => glrmask::Optimization::Balanced,
        PyOptimization::FAST_RUNTIME => glrmask::Optimization::FastRuntime,
    };
    let backend = parser_backend.as_deref().copied().unwrap_or(PyParserBackend::TEMPLATE_DFA);
    glrmask::BuildOptions::default().end_tokens(end_tokens.unwrap_or_default())
        .optimization(mode).parser_backend(backend.into())
}

/// An original token ID with no vocabulary identity. Only ID membership is checked against the target vocabulary; the same ID can refer to different bytes. Use `vocab.tokens([id])` for a vocabulary-bound singleton.
#[pyclass(name = "Token", module = "glrmask", frozen)]
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
#[pyclass(name = "ExactTokens", module = "glrmask", frozen)]
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

#[derive(Clone)]
enum Binding {
    Source(glrmask::Grammar<'static>),
    Prepared(glrmask::PreparedGrammar),
    Token(glrmask::Token),
    Tokens(glrmask::ExactTokens),
}

impl Binding {
    fn extract_for_grammar(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = value.extract::<PyRef<'_, PyGrammar>>() {
            return Ok(Self::Source(value.inner.clone()));
        }
        if let Ok(value) = value.extract::<PyRef<'_, PyToken>>() {
            return Ok(Self::Token(value.inner.clone()));
        }
        if let Ok(value) = value.extract::<PyRef<'_, PyExactTokens>>() {
            return Ok(Self::Tokens(value.inner.clone()));
        }
        Err(PyTypeError::new_err("Grammar.bind accepts Grammar, Token, or ExactTokens"))
    }

    fn extract_for_unlinked(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value)=value.extract::<PyRef<'_,PyPreparedGrammar>>() {
            return Ok(Self::Prepared(value.inner.clone()));
        }

        if let Ok(value) = value.extract::<PyRef<'_, PyToken>>() {
            return Ok(Self::Token(value.inner.clone()));
        }
        if let Ok(value) = value.extract::<PyRef<'_, PyExactTokens>>() {
            return Ok(Self::Tokens(value.inner.clone()));
        }
        Err(PyTypeError::new_err(
            "PreparedGrammar.bind accepts PreparedGrammar, Token, or ExactTokens",
        ))
    }

    fn apply(&self,grammar:&mut glrmask::Grammar<'static>,name:&str)->glrmask::Result<()> {
        match self {
            Self::Source(child)=>grammar.bind(name,child),
            Self::Token(token)=>grammar.bind(name,token.clone()),
            Self::Tokens(tokens)=>grammar.bind(name,tokens.clone()),
            Self::Prepared(_)=>unreachable!("compiled bindings rejected by source bind"),
        }
    }

}

/// Reusable source grammar with mutable bindings. `bind` changes this handle; clones and already attached children keep independent snapshots. Parsing and vocabulary checks can be deferred until preparation or compilation.
#[pyclass(name = "Grammar", module = "glrmask")]
#[derive(Clone)]
pub(super) struct PyGrammar {
    inner:glrmask::Grammar<'static>,
}

impl PyGrammar {
    fn new(source:String,kind:SourceKind)->Self {
        let inner=match kind {
            SourceKind::Ebnf=>glrmask::Grammar::from_ebnf_owned(source),
            SourceKind::Glrm=>glrmask::Grammar::from_glrm_owned(source),
            SourceKind::JsonSchema=>glrmask::Grammar::from_json_schema_owned(source),
        };
        Self {inner}
    }
    fn as_rust(&self)->glrmask::Result<glrmask::Grammar<'static>> {Ok(self.inner.clone())}
}

/// An unresolved typed slot. Paths distinguish the same local name in different attached children.
#[pyclass(name="Requirement",module="glrmask",frozen)]
#[derive(Clone)]
pub(super) struct PyRequirement {inner:glrmask::Requirement}
#[pymethods]
impl PyRequirement {
    /// Qualified path joined with `::`.
    #[getter]
    fn name(&self)->String {self.inner.name()}
    /// Path segments from the assembly root to the unresolved slot.
    #[getter]
    fn path(&self)->Vec<String> {self.inner.path.clone()}
    /// `(start_byte, end_byte, line, column)` in the declaring source, or `None`. The byte interval is half-open; line and Unicode-scalar column are one-based.
    #[getter]
    fn source_span(&self)->Option<(usize,usize,usize,usize)> {
        self.inner.source_span.map(|span|(span.start,span.end,span.line,span.column))
    }
    /// A token slot accepts exact events; a grammar slot also accepts a grammar component.
    #[getter]
    fn kind(&self)->&'static str {match self.inner.kind {
        glrmask::ExternalKind::Token=>"token",glrmask::ExternalKind::Grammar=>"grammar",glrmask::ExternalKind::LegacyGeneric=>"legacy",
    }}
    fn __repr__(&self)->String {format!("Requirement({:?}, kind={:?})",self.name(),self.kind())}
}

#[pymethods]
impl PyGrammar {
    /// Parsing and compilation errors can be deferred until `prepare` or `compile`.
    #[staticmethod]
    fn from_ebnf(source: String) -> Self { Self::new(source, SourceKind::Ebnf) }

    /// The default `recognition` profile and the `llm` extension profile differ in accepted syntax. Named token atoms require explicit `token_names` mappings; they are never inferred from tokenizer spellings.
    #[staticmethod]
    #[pyo3(signature=(source, *, profile="recognition", token_names=None))]
    fn from_lark(source:String,profile:&str,token_names:Option<&Bound<'_,PyDict>>)->PyResult<Self> {
        let mut options=match profile {
            "recognition"=>glrmask::LarkOptions::default(),
            "llm"=>glrmask::LarkOptions::llm(),
            _=>return Err(PyValueError::new_err("Lark profile must be 'recognition' or 'llm'")),
        };
        if let Some(names)=token_names {
            for (name,value) in names.iter() {
                let name:String=name.extract()?;
                let ids=match value.extract::<u32>() {Ok(id)=>vec![id],Err(_)=>value.extract::<Vec<u32>>()?};
                options=options.token(name,ids);
            }
        }
        Ok(Self {inner:glrmask::Grammar::from_lark_owned(source,options)})
    }

    /// Adapt one vocabulary-bound token event to a grammar fragment. Adds no byte spelling, ignore rule or root end-token policy.
    #[staticmethod]
    fn exact(tokens:&PyExactTokens)->Self {Self {inner:glrmask::Grammar::exact(tokens.inner.clone())}}

    /// Includes unresolved slots in attached open children. Diagnostic expansion is bounded and may raise an error for very large shared graphs.
    #[getter]
    fn requirements(slf:&Bound<'_,Self>)->PyResult<Vec<PyRequirement>> {
        let snapshot=slf.try_borrow()?.inner.clone();
        slf.py().allow_threads(move ||snapshot.requirements())
            .map(|values|values.into_iter().map(|inner|PyRequirement{inner}).collect()).map_err(api_error)
    }

    /// GLRM supports typed external slots; fill them with `bind` before compiling a runnable root.
    #[staticmethod]
    fn from_glrm(source: String) -> Self { Self::new(source, SourceKind::Glrm) }

    /// Accepts JSON text or a JSON-serializable object. The importer implements a subset of JSON Schema: unsupported assertions may be silently weakened. Validate generated values separately when full schema conformance matters.
    #[staticmethod]
    fn from_json_schema(schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        let source = match schema.extract::<String>() {
            Ok(source) => source,
            Err(_) => schema.py().import("json")?.getattr("dumps")?.call1((schema,))?.extract()?,
        };
        Ok(Self::new(source, SourceKind::JsonSchema))
    }

    /// Fill a declared slot once. Use `::` for nested paths, after attaching the parent. Token slots require `Token` or `ExactTokens`. Attached children are snapshots; failed binds leave this handle unchanged.
    fn bind(slf:&Bound<'_,Self>,name:String,value:&Bound<'_,PyAny>)->PyResult<()> {
        let py=slf.py();
        // Capture value first: self-attachment captures the old snapshot too.
        let binding=Binding::extract_for_grammar(value)?;
        let mut target=slf.try_borrow_mut()?;
        let mut next=target.inner.clone();
        let next=py.allow_threads(move || {
            binding.apply(&mut next,&name)?;
            Ok::<_,glrmask::Error>(next)
        }).map_err(api_error)?;
        let previous=std::mem::replace(&mut target.inner,next);
        // A last snapshot drop may release cached diagnostic paths. Never hold
        // the GIL through that potentially large native destruction.
        py.allow_threads(move ||drop(previous));
        Ok(())
    }

    /// Independent binding snapshot; later binds on either handle do not affect the other.
    #[pyo3(name="clone")]
    fn clone_handle(&self)->Self {Self {inner:self.inner.clone()}}
    /// Equivalent to `clone`.
    fn __copy__(&self)->Self {self.clone_handle()}
    /// Equivalent to `clone`; immutable compiled storage may remain shared.
    fn __deepcopy__(slf:&Bound<'_,Self>,memo:&Bound<'_,PyDict>)->PyResult<Py<Self>> {
        let copy=Py::new(slf.py(),slf.try_borrow()?.clone_handle())?;
        memo.set_item(slf.as_ptr() as usize,&copy)?;
        Ok(copy)
    }


    /// Requires every slot to be bound. `end_tokens` are root generation controls, allowed only at acceptance; commit the selected end token before stopping. Their policy is not inherited by an embedded child. `optimization` defaults to `AUTO`.
    #[pyo3(signature = (vocab, *, end_tokens=None, optimization=None, boundary_trigger=None))]
    fn compile(
        slf:&Bound<'_,Self>, py: Python<'_>, vocab: &PyVocab,
        end_tokens: Option<Vec<u32>>, optimization: Option<PyRef<'_, PyOptimization>>,
        boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>,
    ) -> PyResult<PyConstraint> {
        let options = with_boundary(options(end_tokens, optimization, None), boundary_trigger);
        let grammar = slf.try_borrow()?.inner.clone();
        let vocab = vocab.inner.clone();
        py.allow_threads(move || grammar.compile_with(&vocab, options))
            .map(constraint).map_err(api_error)
    }

    /// Compile vocabulary-specific machinery while retaining unresolved slots. Reuse it with different bindings, then `link`. Only the boundary metadata request is retained for final link.
    #[pyo3(signature = (vocab, *, boundary_trigger=None))]
    fn prepare(slf:&Bound<'_,Self>, py: Python<'_>, vocab: &PyVocab,
        boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>) -> PyResult<PyPreparedGrammar> {
        let options = with_boundary(glrmask::BuildOptions::default(), boundary_trigger);
        let grammar = slf.try_borrow()?.inner.clone();
        let vocab = vocab.inner.clone();
        py.allow_threads(move || grammar.prepare_with(&vocab, options))
            .map(|inner| PyPreparedGrammar { inner }).map_err(api_error)
    }
}

/// Reusable compiled machinery tied to one vocabulary, possibly with open slots. Bind prepared children or exact events, then link a runnable root. Bindings are mutable on this handle; clones and attached children retain independent snapshots.
#[pyclass(name = "PreparedGrammar", module = "glrmask")]
#[derive(Clone)]
pub(super) struct PyPreparedGrammar {
    inner: glrmask::PreparedGrammar,
}

#[pymethods]
impl PyPreparedGrammar {
    /// Unresolved paths, including slots inside attached children. Diagnostic expansion is bounded.
    #[getter]
    fn requirements(slf:&Bound<'_,Self>)->PyResult<Vec<PyRequirement>> {
        let snapshot=slf.try_borrow()?.inner.clone();
        slf.py().allow_threads(move ||snapshot.requirements())
            .map(|values|values.into_iter().map(|inner|PyRequirement{inner}).collect()).map_err(api_error)
    }
    /// Fill a declared local or qualified slot once with `PreparedGrammar`, `Token` or `ExactTokens`. Prepared children must use the same vocabulary mapping. Captures a child snapshot without recompiling it. Invalid bindings leave this handle unchanged.
    fn bind(slf:&Bound<'_,Self>,py:Python<'_>,name:String,value:&Bound<'_,PyAny>)->PyResult<()> {
        let binding=Binding::extract_for_unlinked(value)?;
        let mut target=slf.try_borrow_mut()?;
        let mut next=target.inner.clone();
        let next=py.allow_threads(move || {
            match binding {
                Binding::Prepared(child)=>next.bind(&name,child)?,
                Binding::Token(token)=>next.bind(&name,token)?,
                Binding::Tokens(tokens)=>next.bind(&name,tokens)?,
                Binding::Source(_)=>unreachable!("source bindings rejected above"),
            }
            Ok::<_,glrmask::Error>(next)
        }).map_err(api_error)?;
        let previous=std::mem::replace(&mut target.inner,next);
        py.allow_threads(move ||drop(previous));
        Ok(())
    }

    /// Independent binding snapshot sharing immutable compiled machinery.
    #[pyo3(name="clone")]
    fn clone_handle(&self)->Self {Self {inner:self.inner.clone()}}
    /// Equivalent to `clone`.
    fn __copy__(&self)->Self {self.clone_handle()}
    /// Equivalent to `clone`; immutable compiled storage may remain shared.
    fn __deepcopy__(slf:&Bound<'_,Self>,memo:&Bound<'_,PyDict>)->PyResult<Py<Self>> {
        let copy=Py::new(slf.py(),slf.try_borrow()?.clone_handle())?;
        memo.set_item(slf.as_ptr() as usize,&copy)?;
        Ok(copy)
    }

    /// Requires all slots to be bound. Selects final optimization and root end tokens; an explicit boundary request overrides the prepared request. End tokens become allowed only at acceptance and must be committed before stopping.
    #[pyo3(signature = (*, end_tokens=None, optimization=None, boundary_trigger=None))]
    fn link(
        slf:&Bound<'_,Self>, py: Python<'_>, end_tokens: Option<Vec<u32>>,
        optimization: Option<PyRef<'_, PyOptimization>>,
        boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>,
    ) -> PyResult<PyConstraint> {
        let options = with_boundary(options(end_tokens, optimization, None), boundary_trigger);
        let snapshot=slf.try_borrow()?.inner.clone();
        py.allow_threads(move || snapshot.link_with(options)).map(constraint).map_err(api_error)
    }

    /// Preserves compiled machinery, vocabulary and unresolved slots. Shared definitions are serialized once.
    fn save<'py>(slf:&Bound<'py,Self>, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let snapshot=slf.try_borrow()?.inner.clone();
        let bytes = py.allow_threads(move || snapshot.save());
        Ok(PyBytes::new(py, &bytes))
    }

    /// Rejects malformed or incompatible artifacts. An optional vocabulary must match the saved token-ID/byte mapping and lets the loaded graph share its caches.
    #[staticmethod]
    #[pyo3(signature = (data, vocab=None))]
    fn load(py: Python<'_>, data: &[u8], vocab: Option<&PyVocab>) -> PyResult<Self> {
        let data = data.to_vec();
        let vocab = vocab.map(|v| v.inner.clone());
        py.allow_threads(move || match vocab {
            Some(vocab) => glrmask::PreparedGrammar::load_with_vocab(data, &vocab),
            None => glrmask::PreparedGrammar::load(data),
        }).map(|inner| Self { inner }).map_err(api_error)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGrammar>()?;
    module.add_class::<PyPreparedGrammar>()?;
    module.add_class::<PyRequirement>()?;
    module.add_class::<PyToken>()?;
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
    module: &PyPreparedGrammar, py: Python<'_>, end_tokens: Option<Vec<u32>>,
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
