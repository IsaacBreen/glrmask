//! Python's source/module/root lifecycle, backed by the production Rust API.
mod canonical;
use super::{check_token_id, PyConstraint, PyVocab};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyModule};

pub(super) fn api_error(error: glrmask::Error) -> PyErr {
    super::glrmask_error(super::kind_name(error.kind()), error.to_string())
}

pub(super) fn constraint(inner: glrmask::Constraint) -> PyConstraint {
    PyConstraint::from_inner(inner)
}

/// Controls how the constraint is compiled.
#[allow(non_camel_case_types)]
#[pyclass(name = "MaskStrategy", module = "glrmask", eq, eq_int, frozen, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum PyMaskStrategy {
    /// The default; currently `FAST_MASKS`.
    AUTO = 0,
    /// Faster compiling, slower masks.
    FAST_COMPILE = 3,
    /// Faster masks, slower compiling.
    FAST_MASKS = 4,
}
#[pymethods]
impl PyMaskStrategy {
    fn __hash__(&self) -> isize { *self as isize }
}
impl From<PyMaskStrategy> for glrmask::MaskStrategy {
    fn from(value: PyMaskStrategy) -> Self {
        match value {
            PyMaskStrategy::AUTO => Self::AUTO,
            PyMaskStrategy::FAST_COMPILE => Self::FAST_COMPILE,
            PyMaskStrategy::FAST_MASKS => Self::FAST_MASKS,
        }
    }
}

/// Internal diagnostic value for the actual parser representation.
#[allow(non_camel_case_types)]
#[pyclass(name = "ParserBackend", module = "glrmask._internal", eq, eq_int, frozen, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum PyParserBackend {
    /// An LR table. Faster to compile, with slower masks. The default.
    LR_TABLE,
    /// Template DFAs. Slower to compile, with faster masks.
    TEMPLATE_DFA,
}

#[pymethods]
impl PyParserBackend {
    fn __hash__(&self) -> isize { *self as isize }
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

pub(super) fn resolve_mask_strategy(value: Option<&Bound<'_, PyAny>>) -> PyResult<glrmask::MaskStrategy> {
    let Some(value) = value else { return Ok(glrmask::MaskStrategy::AUTO); };
    if let Ok(strategy) = value.extract::<PyRef<'_, PyMaskStrategy>>() {
        return Ok((*strategy).into());
    }
    Err(PyTypeError::new_err("mask_strategy must be a MaskStrategy"))
}

/// A set of token IDs.
#[pyclass(name = "TokenSet", module = "glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyTokenSet {
    pub(super) inner: glrmask::TokenSet,
}

fn python_range(value: &Bound<'_, PyAny>, ranges: &mut Vec<std::ops::RangeInclusive<u32>>) -> PyResult<()> {
    let range_type = value.py().import("builtins")?.getattr("range")?;
    if !value.is_instance(&range_type)? {
        return Err(PyTypeError::new_err("TokenSet.from_ranges expects range objects"));
    }
    if !value.is_truthy()? {
        return Ok(());
    }
    let convert = |id: i128| match u32::try_from(id) {
        Ok(id) => check_token_id(id),
        Err(_) => Err(super::glrmask_error("invalid_input", format!(
            "token ID {id} is out of range; token IDs go from 0 to {}",
            glrmask::MAX_TOKEN_ID
        ))),
    };
    // Validate both endpoints before expanding a potentially huge stepped range.
    let first = convert(value.get_item(0)?.extract()?)?;
    let last = convert(value.get_item(-1)?.extract()?)?;
    if value.getattr("step")?.eq(1)? {
        ranges.push(first..=last);
    } else {
        for item in value.try_iter()? {
            let id = convert(item?.extract()?)?;
            ranges.push(id..=id);
        }
    }
    Ok(())
}

#[pymethods]
impl PyTokenSet {
    #[new]
    #[pyo3(signature = (ids, /))]
    fn new(ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ids = ids
            .try_iter()?
            .map(|item| item?.extract::<u32>().and_then(check_token_id))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: glrmask::TokenSet::new(ids),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (ranges, /))]
    fn from_ranges(ranges: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut normalized = Vec::new();
        for item in ranges.try_iter()? {
            python_range(&item?, &mut normalized)?;
        }
        Ok(Self { inner: glrmask::TokenSet::from_ranges(normalized) })
    }

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

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
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
    /// Parses [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md) source.
    #[staticmethod]
    #[pyo3(signature = (source, /))]
    fn from_glrm(source: String) -> PyResult<Self> {
        Self::new(source, SourceKind::Glrm)
    }

    /// Parses a [JSON Schema](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md#grammar-formats), given as JSON text or a dict.
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

    /// Parses [Lark](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md#grammar-formats) source.
    #[staticmethod]
    #[pyo3(signature = (source, /))]
    fn from_lark(source: String) -> PyResult<Self> {
        Self::new(source, SourceKind::Lark)
    }

    /// Builds a grammar from an `Expr`.
    #[staticmethod]
    #[pyo3(signature = (expr, /))]
    fn from_expr(expr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let expr = canonical::extract_expr_part(expr)?;
        Ok(Self {
            inner: glrmask::Grammar::from_expr(expr).map_err(api_error)?,
        })
    }

    /// Binds externs by name. If one fails, none are bound.
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

    #[getter]
    fn externs(&self) -> PyResult<canonical::PyExterns> {
        let inner = self.inner.clone();
        let all: Vec<canonical::PyExtern> = inner.externs().iter().map(canonical::PyExtern::from).collect();
        let unbound: Vec<canonical::PyExtern> = inner.externs().unbound().iter().map(canonical::PyExtern::from).collect();
        Ok(canonical::PyExterns { entries: all, unbound_entries: Some(unbound) })
    }

    /// Compiles the grammar into a constraint.
    #[pyo3(signature = (vocab, *, mask_strategy = None))]
    fn compile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        vocab: &PyVocab,
        mask_strategy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyConstraint> {
        let grammar = slf.try_borrow()?.inner.clone();
        let vocab = vocab.inner.clone();
        let options = canonical::compile_options(mask_strategy)?;
        py.detach(move || {
            let module = glrmask::Module::new(&grammar);
            module.compile_with(&vocab, options)
        })
        .map(constraint)
        .map_err(api_error)
    }

    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

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
    Ok(())
}

/// Tooling-only inspection of the stored parser representation.
#[pyfunction]
fn parser_backend(constraint: &PyConstraint) -> PyResult<PyParserBackend> {
    constraint.inner.__inner().parser_backend().try_into()
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
    let strategy: glrmask::InternalMaskStrategy = resolve_mask_strategy(mask_strategy)?.into();
    let options = glrmask::BuildOptions::default()
        .end_tokens(end_tokens.unwrap_or_default())
        .optimization(strategy)
        .parser_backend(glrmask::ParserBackend::TemplateDfa);
    let grammar = grammar.as_rust().map_err(api_error)?;
    py.detach(|| grammar.compile_with_build_options(&vocab.inner, options))
        .map(glrmask::Constraint::__from_inner)
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
