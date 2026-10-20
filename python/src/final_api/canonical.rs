//! Owned Module/Expr facade. Capture all Python inputs before native work.
use super::*;
use glrmask::{IntoExpr as _, IntoModuleValue as _};

fn python_optimization(value: glrmask::Optimization) -> PyResult<PyOptimization> {
    Ok(match value {
        glrmask::Optimization::Auto => PyOptimization::AUTO,
        glrmask::Optimization::FastBuild => PyOptimization::FAST_BUILD,
        glrmask::Optimization::Balanced => PyOptimization::BALANCED,
        glrmask::Optimization::FastRuntime => PyOptimization::FAST_RUNTIME,
        _ => return Err(PyValueError::new_err("unsupported optimization")),
    })
}
fn python_boundary(value: glrmask::BoundaryTriggerDetail) -> PyBoundaryTriggerDetail {
    match value {
        glrmask::BoundaryTriggerDetail::None => PyBoundaryTriggerDetail::NONE,
        glrmask::BoundaryTriggerDetail::Tokens => PyBoundaryTriggerDetail::TOKENS,
        glrmask::BoundaryTriggerDetail::Exact => PyBoundaryTriggerDetail::EXACT,
    }
}

/// Explicit options replace the value; keywords alone update selected local fields.
/// The two forms are mutually exclusive, matching the constructor's contract.
fn updated_options(
    current: glrmask::ModuleOptions,
    options: Option<PyRef<'_, PyModuleOptions>>,
    optimization: Option<PyRef<'_, PyOptimization>>,
    boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>,
) -> PyResult<glrmask::ModuleOptions> {
    if options.is_some() && (optimization.is_some() || boundary_trigger.is_some()) {
        return Err(PyTypeError::new_err("options cannot be combined with optimization or boundary_trigger overrides"));
    }
    let mut result=options.map(|value|value.inner.clone()).unwrap_or(current);
    if let Some(value)=optimization {result=result.optimization(super::optimization(*value));}
    if let Some(value)=boundary_trigger {result=result.boundary_trigger((*value).into());}
    Ok(result)
}

/// Immutable snapshot of component-local optimization and boundary metadata settings.
#[pyclass(name="ModuleOptions", module="glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyModuleOptions { inner: glrmask::ModuleOptions }
#[pymethods]
impl PyModuleOptions {
    #[new]
    #[pyo3(signature=(*, optimization=None, boundary_trigger=None))]
    fn new(optimization: Option<PyRef<'_, PyOptimization>>, boundary_trigger: Option<PyRef<'_, PyBoundaryTriggerDetail>>) -> Self {
        let mut inner=glrmask::ModuleOptions::default();
        if let Some(value)=optimization {inner=inner.optimization(super::optimization(*value));}
        if let Some(value)=boundary_trigger {inner=inner.boundary_trigger((*value).into());}
        Self {inner}
    }
    #[getter]
    fn optimization(&self)->PyResult<PyOptimization> {python_optimization(self.inner.optimization_value())}
    #[getter]
    fn boundary_trigger(&self)->PyBoundaryTriggerDetail {python_boundary(self.inner.boundary_trigger_value())}
    fn __eq__(&self, other: &Self)->bool {self.inner==other.inner}
    fn __repr__(&self)->String {format!("ModuleOptions(optimization={:?}, boundary_trigger={:?})",self.inner.optimization_value(),self.inner.boundary_trigger_value())}
}

/// Immutable root generation settings. Module.compile accepts these as end_tokens kwargs.
#[pyclass(name="CompileOptions", module="glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyCompileOptions { end_tokens: Vec<u32> }
#[pymethods]
impl PyCompileOptions {
    #[new]
    #[pyo3(signature=(*, end_tokens=None))]
    fn new(end_tokens: Option<Vec<u32>>)->Self {Self {end_tokens:end_tokens.unwrap_or_default()}}
    #[getter]
    fn end_tokens(&self)->Vec<u32> {self.end_tokens.clone()}
}

/// Immutable union or concatenation of independently scoped Grammar, Module or Expr snapshots.
#[pyclass(name="Expr", module="glrmask", frozen, from_py_object)]
#[derive(Clone)]
pub(super) struct PyExpr { pub(super) inner: glrmask::Expr }
impl PyExpr {
    fn extract(value: &Bound<'_,PyAny>)->PyResult<glrmask::Expr> {
        if let Ok(value)=value.extract::<PyRef<'_,PyGrammar>>() {return Ok(value.inner.clone().into_expr());}
        if let Ok(value)=value.extract::<PyRef<'_,PyModule>>() {return Ok(value.snapshot().into_expr());}
        if let Ok(value)=value.extract::<PyRef<'_,PyExpr>>() {return Ok(value.inner.clone());}
        Err(PyTypeError::new_err("Expr operands must be Grammar, Module, or Expr"))
    }
    fn collect(values: &Bound<'_,PyAny>)->PyResult<Vec<glrmask::Expr>> {
        // Iteration may invoke Python. No source borrow survives an iteration.
        let mut parts=Vec::new();
        for value in values.try_iter()? {
            match value.and_then(|value|Self::extract(&value)) {
                Ok(value)=>parts.push(value),
                Err(error)=>{values.py().detach(move ||drop(parts));return Err(error);}
            }
        }
        Ok(parts)
    }
}
impl Drop for PyExpr {
    fn drop(&mut self) {
        let inner=std::mem::replace(&mut self.inner,glrmask::Expr::sequence(Vec::<glrmask::Expr>::new()));
        let _=Python::try_attach(move |py|py.detach(move ||drop(inner)));
    }
}
#[pymethods]
impl PyExpr {
    /// Unordered union. Empty choice is rejected when materialized as Grammar or Module.
    #[staticmethod]
    fn choice(values: &Bound<'_,PyAny>)->PyResult<Self> {Ok(Self {inner:glrmask::Expr::choice(Self::collect(values)?)})}
    /// Concatenation, including tokens crossing leaf boundaries. Empty sequence is epsilon.
    #[staticmethod]
    fn sequence(values: &Bound<'_,PyAny>)->PyResult<Self> {Ok(Self {inner:glrmask::Expr::sequence(Self::collect(values)?)})}
}

/// Owned mutable assembly. Options apply locally; children keep their own snapshots and options.
#[pyclass(name="Module", module="glrmask", from_py_object)]
#[derive(Clone)]
pub(super) struct PyModule { inner: Option<glrmask::Module> }
impl PyModule {
    pub(super) fn from_inner(inner: glrmask::Module)->Self {Self {inner:Some(inner)}}
    fn snapshot(&self)->glrmask::Module {self.inner.as_ref().expect("live Module").clone()}
    fn extract(value: &Bound<'_,PyAny>)->PyResult<glrmask::ModuleValue> {
        if let Ok(value)=value.extract::<PyRef<'_,PyGrammar>>() {return value.inner.clone().into_module_value().map_err(api_error);}
        if let Ok(value)=value.extract::<PyRef<'_,PyModule>>() {return value.snapshot().into_module_value().map_err(api_error);}
        if let Ok(value)=value.extract::<PyRef<'_,PyExpr>>() {
            let expr=value.inner.clone();let py=value.py();drop(value);
            return py.detach(move ||expr.into_module_value()).map_err(api_error);
        }
        if let Ok(value)=value.extract::<PyRef<'_,PyToken>>() {return value.inner.into_module_value().map_err(api_error);}
        if let Ok(value)=value.extract::<PyRef<'_,PyExactTokens>>() {return value.inner.clone().into_module_value().map_err(api_error);}
        Err(PyTypeError::new_err("Module.bind accepts Grammar, Module, Expr, Token, or ExactTokens"))
    }
    fn replace(&mut self, py:Python<'_>, next: glrmask::Module) {
        let previous=self.inner.replace(next);py.detach(move ||drop(previous));
    }
}
impl Drop for PyModule {
    fn drop(&mut self) {
        if let Some(inner)=self.inner.take() {
            // During interpreter shutdown attachment may be unavailable; then
            // the closure and its captured native owner are dropped normally.
            let _=Python::try_attach(move |py|py.detach(move ||drop(inner)));
        }
    }
}
#[pymethods]
impl PyModule {
    #[new]
    #[pyo3(signature=(grammar_or_expr, *, options=None, optimization=None, boundary_trigger=None))]
    fn new(grammar_or_expr:&Bound<'_,PyAny>, options:Option<PyRef<'_,PyModuleOptions>>, optimization:Option<PyRef<'_,PyOptimization>>, boundary_trigger:Option<PyRef<'_,PyBoundaryTriggerDetail>>)->PyResult<Self> {
        if options.is_some() && (optimization.is_some() || boundary_trigger.is_some()) {
            return Err(PyTypeError::new_err("options cannot be combined with optimization or boundary_trigger overrides"));
        }
        let options=options.map(|o|o.inner.clone()).unwrap_or_else(||PyModuleOptions::new(optimization,boundary_trigger).inner);
        let inner=if let Ok(grammar)=grammar_or_expr.extract::<PyRef<'_,PyGrammar>>() {
            glrmask::Module::new(&grammar.inner)
        } else if let Ok(expr)=grammar_or_expr.extract::<PyRef<'_,PyExpr>>() {
            let expr=expr.inner.clone();
            grammar_or_expr.py().detach(move ||glrmask::Module::from_expr(&expr)).map_err(api_error)?
        } else {return Err(PyTypeError::new_err("Module expects Grammar or Expr"));};
        grammar_or_expr.py().detach(move ||inner.with_options(options)).map(Self::from_inner).map_err(api_error)
    }
    #[getter]
    fn options(&self)->PyModuleOptions {PyModuleOptions {inner:self.inner.as_ref().unwrap().options().clone()}}
    #[pyo3(signature=(options=None, *, optimization=None, boundary_trigger=None))]
    fn with_options(slf:&Bound<'_,Self>, options:Option<PyRef<'_,PyModuleOptions>>,
        optimization:Option<PyRef<'_,PyOptimization>>, boundary_trigger:Option<PyRef<'_,PyBoundaryTriggerDetail>>)->PyResult<Self> {
        let snapshot=slf.try_borrow()?.snapshot();
        let options=updated_options(snapshot.options().clone(),options,optimization,boundary_trigger)?;
        slf.py().detach(move ||snapshot.with_options(options)).map(Self::from_inner).map_err(api_error)
    }
    #[pyo3(signature=(options=None, *, optimization=None, boundary_trigger=None))]
    fn set_options(slf:&Bound<'_,Self>, options:Option<PyRef<'_,PyModuleOptions>>,
        optimization:Option<PyRef<'_,PyOptimization>>, boundary_trigger:Option<PyRef<'_,PyBoundaryTriggerDetail>>)->PyResult<()> {
        let mut target=slf.try_borrow_mut()?;let snapshot=target.snapshot();
        let options=updated_options(snapshot.options().clone(),options,optimization,boundary_trigger)?;
        let next=slf.py().detach(move ||snapshot.with_options(options)).map_err(api_error)?;
        target.replace(slf.py(),next);Ok(())
    }
    /// Fill a slot atomically. Self attachment captures the old snapshot before exclusive borrowing.
    fn bind(slf:&Bound<'_,Self>, path:String, value:&Bound<'_,PyAny>)->PyResult<()> {
        let value=Self::extract(value)?;
        let mut target=slf.try_borrow_mut()?;let mut next=target.snapshot();
        let next=slf.py().detach(move ||{next.bind(path,value)?;Ok::<_,glrmask::Error>(next)}).map_err(api_error)?;
        target.replace(slf.py(),next);Ok(())
    }
    /// Prepare missing work using each node's stored local options. Mutates and returns None.
    fn precompile(slf:&Bound<'_,Self>, vocab:&PyVocab)->PyResult<()> {
        let vocab=vocab.inner.clone();let mut target=slf.try_borrow_mut()?;let mut next=target.snapshot();
        let next=slf.py().detach(move ||{next.precompile(&vocab)?;Ok::<_,glrmask::Error>(next)}).map_err(api_error)?;
        target.replace(slf.py(),next);Ok(())
    }
    /// Compile missing work and finalize root end-token settings. Local options cannot be overridden here.
    #[pyo3(signature=(vocab, *, options=None, end_tokens=None))]
    fn compile(slf:&Bound<'_,Self>, vocab:&PyVocab, options:Option<PyRef<'_,PyCompileOptions>>, end_tokens:Option<Vec<u32>>)->PyResult<PyConstraint> {
        if options.is_some() && end_tokens.is_some() {
            return Err(PyTypeError::new_err("options cannot be combined with end_tokens"));
        }
        let end_tokens=options.map(|value|value.end_tokens.clone()).or(end_tokens).unwrap_or_default();
        let snapshot=slf.try_borrow()?.snapshot();let vocab=vocab.inner.clone();
        let options=glrmask::CompileOptions::default().end_tokens(end_tokens);
        slf.py().detach(move ||snapshot.compile_with(&vocab,options)).map(constraint).map_err(api_error)
    }
    #[getter]
    fn unbound(slf:&Bound<'_,Self>)->PyResult<Vec<PyRequirement>> {
        let snapshot=slf.try_borrow()?.snapshot();
        slf.py().detach(move ||snapshot.unbound()).map(wrap_externs).map_err(api_error)
    }
    #[getter]
    fn externs(slf:&Bound<'_,Self>)->PyResult<Vec<PyRequirement>> {
        let snapshot=slf.try_borrow()?.snapshot();
        slf.py().detach(move ||snapshot.externs()).map(wrap_externs).map_err(api_error)
    }
    #[pyo3(name="clone")]
    fn clone_handle(&self)->Self {Self::from_inner(self.snapshot())}
    fn __copy__(&self)->Self {self.clone_handle()}
    fn __deepcopy__(slf:&Bound<'_,Self>,memo:&Bound<'_,PyDict>)->PyResult<Py<Self>> {
        let copy=Py::new(slf.py(),slf.try_borrow()?.clone_handle())?;
        memo.set_item(slf.as_ptr() as usize,&copy)?;Ok(copy)
    }
    fn save<'py>(slf:&Bound<'py,Self>)->PyResult<Bound<'py,PyBytes>> {
        let snapshot=slf.try_borrow()?.snapshot();let bytes=slf.py().detach(move ||snapshot.save());
        Ok(PyBytes::new(slf.py(),&bytes))
    }
    /// Load saved source and work without compiling. An optional vocabulary is validated exactly.
    #[staticmethod]
    #[pyo3(signature=(data, vocab=None))]
    fn load(py:Python<'_>, data:&[u8], vocab:Option<&PyVocab>)->PyResult<Self> {
        let data=data.to_vec();let vocab=vocab.map(|v|v.inner.clone());
        py.detach(move ||match vocab {
            Some(vocab)=>glrmask::Module::load_with_vocab(data,&vocab),
            None=>glrmask::Module::load(data),
        }).map(Self::from_inner).map_err(api_error)
    }
}
fn wrap_externs(values:Vec<glrmask::UnboundExtern>)->Vec<PyRequirement> {values.into_iter().map(|inner|PyRequirement {inner}).collect()}

pub(super) fn register(module:&Bound<'_,pyo3::types::PyModule>)->PyResult<()> {
    module.add_class::<PyModule>()?;module.add_class::<PyModuleOptions>()?;
    module.add_class::<PyCompileOptions>()?;module.add_class::<PyExpr>()?;Ok(())
}
