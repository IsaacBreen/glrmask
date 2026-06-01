// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _glrmask(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVocab>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyConstraintState>()?;
    m.add_function(wrap_pyfunction!(clear_stale_weights, m)?)?;
    m.add_function(wrap_pyfunction!(clear_weight_interners, m)?)?;
    m.add_function(wrap_pyfunction!(clear_weight_op_caches, m)?)?;
    m.add_function(wrap_pyfunction!(compile_grammar_def_json, m)?)?;
    m.add_function(wrap_pyfunction!(dump_json_schema_grammar_glrm, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_vocab_for_compile, m)?)?;
    Ok(())
}

#[pyfunction]
fn clear_stale_weights() {
    glrmask::diagnostics::cache::clear_stale_weights();
}

#[pyfunction]
fn clear_weight_interners() {
    glrmask::diagnostics::cache::clear_weight_interners();
}

#[pyfunction]
fn clear_weight_op_caches() {
    glrmask::diagnostics::cache::clear_weight_op_caches();
}

#[pyfunction]
fn prepare_vocab_for_compile(vocab: &PyVocab) {
    glrmask::diagnostics::frontend::prepare_vocab_for_compile(&vocab.inner);
}

#[pyfunction]
fn compile_grammar_def_json(grammar_def_json: &str, vocab: &PyVocab) -> PyResult<PyConstraint> {
    let constraint = glrmask::diagnostics::frontend::compile_grammar_def_json(grammar_def_json, &vocab.inner)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let max_token = vocab.inner.max_token_id();
    Ok(PyConstraint {
        inner: std::sync::Arc::new(constraint),
        max_token,
    })
}

#[pyfunction]
fn dump_json_schema_grammar_glrm(schema_json: &str) -> PyResult<String> {
    glrmask::diagnostics::frontend::dump_json_schema_grammar_glrm(schema_json)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}
