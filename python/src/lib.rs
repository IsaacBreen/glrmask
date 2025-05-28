use sep1::tokenizer::LLMTokenID;
use sep1::finite_automata::{Expr as RegexExpr, ExprGroups as RegexGroups, greedy_group, non_greedy_group, groups as regex_groups, _choice as regex_choice, eat_u8, eat_u8_negation, eat_u8_set, eps, opt, prec, rep, rep1, _seq as regex_seq, ExprGroups};
use sep1::finite_automata::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyDict};
use sep1::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use sep1::glr::parser::{GLRParser, GLRParserState};
use sep1::glr::table::{generate_glr_parser, StateID, TerminalID};
use sep1::interface::{CompiledGrammar, GrammarExpr, choice as grammar_choice, literal as grammar_literal, optional as grammar_optional, regex as grammar_regex, repeat as grammar_repeat, r#ref as grammar_ref, sequence as grammar_sequence, eat_any_fast, GrammarDefinition};
use sep1::constraint::{GrammarConstraint, GrammarConstraintState};
use std::collections::{BTreeMap, BTreeSet};
use bimap::BiBTreeMap;
use std::sync::Arc;
use ouroboros::self_referencing;
use numpy::{IntoPyArray, PyArray1, ToPyArray};
use sep1::interface::IncrementalParser;
use sep1::json_serialization::{JSONConvertible, JSONNode};

#[pyclass(name = "GrammarExpr")]
#[derive(Clone)]
struct PyGrammarExpr {
    inner: GrammarExpr,
}

#[pymethods]
impl PyGrammarExpr {
    #[staticmethod]
    fn r#ref(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: grammar_ref(name),
        })
    }

    #[staticmethod]
    fn sequence(exprs: Vec<PyGrammarExpr>) -> Self {
        Self {
            inner: grammar_sequence(exprs.into_iter().map(|e| e.inner).collect()),
        }
    }

    #[staticmethod]
    fn choice(exprs: Vec<PyGrammarExpr>) -> Self {
        Self {
            inner: grammar_choice(exprs.into_iter().map(|e| e.inner).collect()),
        }
    }

    #[staticmethod]
    fn optional(expr: PyGrammarExpr) -> Self {
        Self {
            inner: grammar_optional(expr.inner),
        }
    }

    #[staticmethod]
    fn repeat(expr: PyGrammarExpr) -> Self {
        Self {
            inner: grammar_repeat(expr.inner),
        }
    }

    #[staticmethod]
    fn regex(regex: PyRegexExpr) -> Self {
        Self {
            inner: grammar_regex(regex.inner)
        }
    }

    #[staticmethod]
    fn literal(bytes: Vec<u8>) -> Self {
        Self {
            inner: grammar_literal(bytes),
        }
    }
}

#[pyclass(name = "RegexExpr")]
#[derive(Clone)]
struct PyRegexExpr {
    inner: RegexExpr,
}

#[pymethods]
impl PyRegexExpr {
    #[staticmethod]
    fn eat_u8(c: u8) -> Self {
        Self { inner: eat_u8(c) }
    }

    #[staticmethod]
    fn eat_u8_negation(c: u8) -> Self {
        Self { inner: eat_u8_negation(c) }
    }

    #[staticmethod]
    pub fn eat_any() -> Self {
        Self { inner: eat_any_fast() }
    }

    #[staticmethod]
    fn rep(expr: PyRegexExpr) -> Self {
        Self { inner: rep(expr.inner) }
    }

    #[staticmethod]
    fn rep1(expr: PyRegexExpr) -> Self {
        Self { inner: rep1(expr.inner) }
    }

    #[staticmethod]
    fn opt(expr: PyRegexExpr) -> Self {
        Self { inner: opt(expr.inner) }
    }

    #[staticmethod]
    fn prec(precedence: isize, expr: PyRegexExpr) -> PyRegexGroup {
        PyRegexGroup { inner: prec(precedence, expr.inner) }
    }

    #[staticmethod]
    fn eps() -> Self {
        Self { inner: eps() }
    }

    #[staticmethod]
    fn seq(exprs: Vec<PyRegexExpr>) -> Self {
        Self { inner: regex_seq(exprs.into_iter().map(|e| e.inner).collect()) }
    }

    #[staticmethod]
    fn choice(exprs: Vec<PyRegexExpr>) -> Self {
        Self { inner: regex_choice(exprs.into_iter().map(|e| e.inner).collect()) }
    }

    fn build(&self) -> PyResult<PyRegex> {
        Ok(PyRegex { inner: self.inner.clone().build() })
    }
}

#[pyclass(name = "RegexGroup")]
#[derive(Clone)]
struct PyRegexGroup {
    inner: sep1::finite_automata::ExprGroup,
}

#[pymethods]
impl PyRegexGroup {
    #[staticmethod]
    fn greedy_group(expr: PyRegexExpr) -> Self {
        Self { inner: greedy_group(expr.inner) }
    }

    #[staticmethod]
    fn non_greedy_group(expr: PyRegexExpr) -> Self {
        Self { inner: non_greedy_group(expr.inner) }
    }
}

#[pyclass(name = "RegexGroups")]
#[derive(Clone)]
struct PyRegexGroups {
    inner: RegexGroups,
}

#[pymethods]
impl PyRegexGroups {
    #[staticmethod]
    fn groups(groups: Vec<PyRegexGroup>) -> Self {
        Self {
            inner: regex_groups(groups.into_iter().map(|g| g.inner).collect()),
        }
    }

    fn build(&self) -> PyResult<PyRegex> {
        Ok(PyRegex { inner: self.inner.clone().build() })
    }
}

#[pyclass(name = "Regex")]
#[derive(Clone)]
pub struct PyRegex {
    inner: Regex,
}

#[pymethods]
impl PyRegex {
    // Python methods for PyRegex if needed
}

#[pyclass(name = "GrammarDefinition")]
#[derive(Clone)]
pub struct PyGrammarDefinition {
    inner: GrammarDefinition,
}

#[pymethods]
impl PyGrammarDefinition {
    #[new]
    fn new(exprs: Vec<(String, PyGrammarExpr)>) -> PyResult<Self> {
        let inner_exprs = exprs.into_iter().map(|(s, e)| (s, e.inner)).collect();
        let compiled_grammar = GrammarDefinition::from_exprs(inner_exprs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to compile grammar: {}", e)))?;
        Ok(PyGrammarDefinition { inner: compiled_grammar })
    }

    fn simplify(&mut self) {
        self.inner.simplify();
    }

    fn compile(&self) -> PyResult<PyCompiledGrammar> {
        let compiled_grammar = CompiledGrammar::from_definition(Arc::new(self.inner.clone()));
        Ok(PyCompiledGrammar { inner: compiled_grammar })
    }

    fn print(&self) {
        // The Debug impl for GrammarDefinition is quite verbose.
        // Consider a more Python-friendly summary or selective printing.
        println!("{}", self.inner);
    }

    fn to_json_string(&self) -> PyResult<String> {
        Ok(self.inner.to_json().to_json_string())
    }

    #[staticmethod]
    fn from_json_string(json_str: &str) -> PyResult<Self> {
        let json_node = JSONNode::from_json_string(json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse JSON string to JSONNode: {}", e)))?;
        let grammar = GrammarDefinition::from_json(json_node)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to deserialize GrammarDefinition from JSONNode: {}", e)))?;
        Ok(PyGrammarDefinition { inner: grammar })
    }
}


#[pyclass(name = "CompiledGrammar")]
#[derive(Clone)]
pub struct PyCompiledGrammar {
    inner: CompiledGrammar,
}

#[pymethods]
impl PyCompiledGrammar {
    #[new]
    fn new(exprs: Vec<(String, PyGrammarExpr)>) -> PyResult<Self> {
        let inner_exprs = exprs.into_iter().map(|(s, e)| (s, e.inner)).collect();
        let compiled_grammar = CompiledGrammar::from_exprs(inner_exprs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to compile grammar: {}", e)))?;
        Ok(Self { inner: compiled_grammar })
    }

    // If direct access to GLRParser is needed in Python, expose it.
    // fn glr_parser(&self) -> PyGLRParser {
    //     PyGLRParser { inner: self.inner.glr_parser().clone() } // Clone if GLRParser is Clone
    // }

    fn print(&self) {
        // The Debug impl for CompiledGrammar is quite verbose.
        // Consider a more Python-friendly summary or selective printing.
        println!("{}", self.inner);
    }

    fn to_json_string(&self) -> PyResult<String> {
        Ok(self.inner.to_json().to_json_string())
    }

    #[staticmethod]
    fn from_json_string(json_str: &str) -> PyResult<Self> {
        let json_node = JSONNode::from_json_string(json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse JSON string to JSONNode: {}", e)))?;
        let grammar = CompiledGrammar::from_json(json_node)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to deserialize CompiledGrammar from JSONNode: {}", e)))?;
        Ok(PyCompiledGrammar { inner: grammar })
    }
}

// PyGLRParser might not be needed if not directly manipulated from Python,
// as it's part of CompiledGrammar.
// #[pyclass]
// #[derive(Clone)]
// pub struct PyGLRParser {
//     inner: GLRParser,
// }
// #[pymethods]
// impl PyGLRParser {
//     fn print(&self) {
//         println!("{}", self.inner)
//     }
// }

#[pyclass(name = "GrammarConstraint")]
#[derive(Clone)]
pub struct PyGrammarConstraint {
    inner: Arc<GrammarConstraint>,
}

#[pymethods]
impl PyGrammarConstraint {
    #[new]
    fn new(py: Python, grammar: PyCompiledGrammar, token_to_id: &Bound<'_, PyDict>, max_llm_token_id: usize) -> PyResult<Self> {
        let mut llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID> = BiBTreeMap::new();
        for (key, value) in token_to_id.iter() {
            let token = key.extract::<&[u8]>()?;
            let id = value.extract::<usize>()?;
            llm_token_map.insert(token.to_vec(), LLMTokenID(id));
        }

        // GrammarConstraint::from_compiled_grammar expects an owned CompiledGrammar.
        // PyCompiledGrammar holds an owned CompiledGrammar, so we clone it.
        // The _eof_llm_token_id is not directly used by GrammarConstraint::new,
        // but it's part of the conceptual model for token ranges.
        // We can pass max_llm_token_id + 1 or a dedicated EOF marker if needed by constraint logic.
        // For now, the Rust API for GrammarConstraint::new doesn't take eof_llm_token_id.
        // The old from_grammar took it, but new from_compiled_grammar doesn't.
        // Let's assume eof handling is implicit or managed by GrammarConstraintState.
        let constraint = GrammarConstraint::from_compiled_grammar(
            grammar.inner.clone(), // Clone the CompiledGrammar
            llm_token_map,
            0, // Placeholder for eof_llm_token_id, as current from_compiled_grammar doesn't take it
            max_llm_token_id,
        );

        Ok(Self { inner: Arc::new(constraint) })
    }

    fn print(&self) {
        // Printing GrammarConstraint can be complex.
        // Consider what information is useful to expose.
        println!("PyGrammarConstraint (details not implemented for print)");
    }

    fn to_json_string(&self) -> PyResult<String> {
        Ok(self.inner.to_json().to_json_string())
    }

    #[staticmethod]
    fn from_json_string(json_str: &str) -> PyResult<Self> {
        let json_node = JSONNode::from_json_string(json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to parse JSON string to JSONNode: {}", e)))?;
        let constraint = GrammarConstraint::from_json(json_node)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to deserialize GrammarConstraint from JSONNode: {}", e)))?;
        Ok(Self { inner: Arc::new(constraint) })
    }
}


#[self_referencing]
struct PyGrammarConstraintStateWrapper {
    constraint: PyGrammarConstraint, // Owns the Arc'd constraint
    #[borrows(constraint)]
    #[covariant]
    inner: GrammarConstraintState<'this>,
}

#[pyclass(name = "GrammarConstraintState")]
pub struct PyGrammarConstraintState {
    inner: PyGrammarConstraintStateWrapper,
}

#[pymethods]
impl PyGrammarConstraintState {
    #[new]
    fn new(constraint: PyGrammarConstraint) -> PyResult<Self> {
        Ok(PyGrammarConstraintState {
            inner: PyGrammarConstraintStateWrapperTryBuilder {
                constraint,
                inner_builder: |c: &PyGrammarConstraint| {
                    let mut state = c.inner.init();
                    state.step_with_all_llm_tokens();
                    Ok::<_, PyErr>(state)
                },
            }
            .try_build()?
        })
    }

    fn get_mask<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bitset = self.inner.with_inner_mut(|state| state.get_mask());
        let bools: Vec<bool> = bitset.iter_bits().collect();
        Ok(bools.into_pyarray_bound(py))
    }

    fn commit(&mut self, llm_token_id: usize) {
        // println!("Committing token {} to grammar constraint state", llm_token_id); // Debug
        self.inner.with_inner_mut(|state| {
            state.commit(LLMTokenID(llm_token_id));
            state.step_with_all_llm_tokens();
        });
    }
}

#[self_referencing]
struct PyIncrementalParserWrapper {
    grammar: PyCompiledGrammar, // Owns the PyCompiledGrammar (which owns CompiledGrammar)
    #[borrows(grammar)]
    #[covariant]
    parser: IncrementalParser<'this>,
}

#[pyclass(name = "IncrementalParser")]
pub struct PyIncrementalParser {
    inner: PyIncrementalParserWrapper,
}

#[pymethods]
impl PyIncrementalParser {
    #[new]
    fn new(grammar: PyCompiledGrammar) -> PyResult<Self> {
        Ok(PyIncrementalParser {
            inner: PyIncrementalParserWrapperTryBuilder {
                grammar, // PyCompiledGrammar is moved in
                parser_builder: |g: &PyCompiledGrammar| Ok::<_, PyErr>(IncrementalParser::new(&g.inner)),
            }.try_build()?
        })
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.inner.with_parser_mut(|p| p.feed(bytes));
    }

    fn is_valid(&self) -> bool {
        self.inner.with_parser(|p| p.is_valid())
    }
}

#[pymodule]
fn _sep1(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrammarExpr>()?;
    m.add_class::<PyRegexExpr>()?;
    m.add_class::<PyRegexGroup>()?;
    m.add_class::<PyRegexGroups>()?;
    m.add_class::<PyRegex>()?;
    m.add_class::<PyCompiledGrammar>()?;
    // m.add_class::<PyGLRParser>()?; // Not exposed directly for now
    m.add_class::<PyGrammarConstraint>()?;
    m.add_class::<PyGrammarConstraintState>()?;
    m.add_class::<PyIncrementalParser>()?;
    Ok(())
}

