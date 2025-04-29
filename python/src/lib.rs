use sep1::tokenizer::LLMTokenID;
use sep1::finite_automata::{Expr as RegexExpr, ExprGroups as RegexGroups, greedy_group, non_greedy_group, groups as regex_groups, _choice as regex_choice, eat_u8, eat_u8_negation, eat_u8_set, eps, opt, prec, rep, rep1, _seq as regex_seq};
use sep1::finite_automata::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyDict};
use sep1::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use sep1::glr::parser::GLRParser;
use sep1::glr::table::{generate_glr_parser, StateID};
use sep1::interface::{Grammar, GrammarExpr, choice as grammar_choice, optional as grammar_optional, regex as grammar_regex, repeat as grammar_repeat, r#ref as grammar_ref, sequence as grammar_sequence};
use sep1::constraint::{GrammarConstraint, GrammarConstraintState};
use std::collections::{BTreeMap, BTreeSet};
use bimap::BiBTreeMap;
use std::sync::Arc;
use ouroboros::self_referencing;
use numpy::{IntoPyArray, PyArray1, ToPyArray};

#[pyclass]
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
}

#[pyclass]
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

#[pyclass]
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

#[pyclass]
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

#[pyclass]
#[derive(Clone)]
pub struct PyRegex {
    inner: Regex,
}

#[pymethods]
impl PyRegex {
}


#[pyclass]
#[derive(Clone)]
pub struct PyGrammar {
    inner: Grammar,
}

#[pymethods]
impl PyGrammar {
    #[new]
    fn new(exprs: Vec<(String, PyGrammarExpr)>) -> Self {
        let inner = Grammar::from_exprs(exprs.into_iter().map(|(s, e)| (s, e.inner)).collect());
        Self { inner }
    }

    fn glr_parser(&self) -> PyGLRParser {
        PyGLRParser { inner: self.inner.glr_parser() }
    }

    fn print(&self) {
        println!("{:?}", self.inner)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyGLRParser {
    inner: GLRParser,
}

#[pyclass]
#[derive(Clone)]
pub struct PyGrammarConstraint {
    inner: Arc<GrammarConstraint>,
}

#[pymethods]
impl PyGrammarConstraint {
    #[new]
    fn new(py: Python, grammar: PyGrammar, token_to_id: &Bound<'_, PyDict>, max_llm_token_id: usize) -> PyResult<Self> {
        let mut llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID> = BiBTreeMap::new();
        for (key, value) in token_to_id.iter() {
            let token = key.extract::<&[u8]>()?;
            let id = value.extract::<usize>()?;
            llm_token_map.insert(token.to_vec(), LLMTokenID(id));
        }

        // Assuming Grammar has methods to get tokenizer and parser
        // You might need to adjust this based on your actual Grammar implementation
        let tokenizer = grammar.inner.tokenizer.clone(); // Placeholder
        let parser = grammar.inner.glr_parser(); // Placeholder

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, max_llm_token_id);
        Ok(Self { inner: Arc::new(constraint) })
    }

    fn print(&self) {
        // Assuming a function like print_precomputed exists and takes &Precomputed
        // Or implement a method on GrammarConstraint to print/dump its state
        // print_precomputed(&self.inner.precomputed);
        println!("Printing precomputed data is not implemented in this binding yet.");
    }
}


#[pyclass]
#[self_referencing]
pub struct PyGrammarConstraintState {
    // Owns the Arc'd constraint
    constraint: PyGrammarConstraint,
    // Borrows from the owned constraint via the 'this lifetime
    #[borrows(constraint)]
    #[covariant] // Allows the lifetime 'this to be shortened
    inner: GrammarConstraintState<'this>,
}

#[pymethods]
impl PyGrammarConstraintState {
    // #[new]
    // fn new(constraint: PyGrammarConstraint) -> PyResult<Self> {
    //     // Use the builder provided by ouroboros
    //     PyGrammarConstraintStateTryBuilder {
    //         constraint,
    //         inner_builder: |constraint: &PyGrammarConstraint| Ok::<_, PyErr>(constraint.inner.init()),
    //     }.try_build()
    // }

    fn get_mask<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bitset = self.with_inner_mut(|state| state.get_mask());
        let bools: Vec<bool> = bitset.iter().map(|bit_ref| *bit_ref).collect();
        let array = bools.into_pyarray_bound(py);
        Ok(array)
    }

    fn commit(&mut self, llm_token_id: usize) {
        self.with_inner_mut(|state| state.commit(LLMTokenID(llm_token_id)));
    }
}

#[pymodule]
fn _sep1(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrammarExpr>()?;
    m.add_class::<PyRegexExpr>()?;
    m.add_class::<PyRegexGroup>()?;
    m.add_class::<PyRegexGroups>()?;
    m.add_class::<PyGrammar>()?;
    m.add_class::<PyGrammarConstraint>()?;
    m.add_class::<PyGrammarConstraintState>()?;
    Ok(())
}