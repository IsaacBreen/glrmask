#![recursion_limit = "256"]

use std::collections::BTreeMap;
use numpy::{IntoPyArray, PyArray1, PyReadwriteArray1};
use ouroboros::self_referencing;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyIterator, PySet, PyTuple};
use sep1::constraint::{GrammarConstraint, GrammarConstraintState, StageVocab};
use sep1::datastructures::bitset::{Bitset as RustBitset, Bitset};
use sep1::datastructures::gss_acc::{Acc as RustAcc, Acc};
use sep1::datastructures::hybrid_bitset::{RangeSet as RustHybridBitset, RangeSet};
use sep1::datastructures::leveled_gss::LeveledGSS;
use sep1::datastructures::u8set::U8Set;
use sep1::finite_automata::Regex;
use sep1::finite_automata::{
    _choice as regex_choice, _seq as regex_seq, eat_u8, eat_u8_negation, eat_u8_seq, eat_u8_set,
    eat_u8_set_negation, eps, greedy_group, groups as regex_groups, non_greedy_group, opt, prec,
    rep, rep1, Expr as RegexExpr, ExprGroups as RegexGroups,
};
use sep1::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateEdgeContent};
use sep1::interface::IncrementalParser;
use sep1::interface::{
    choice as grammar_choice, eat_any_fast, literal as grammar_literal, optional as grammar_optional,
    r#ref as grammar_ref, repeat as grammar_repeat, sequence as grammar_sequence, CompiledGrammar,
    GrammarDefinition, GrammarExpr,
};
use sep1::json_schema::{json_schema_to_ebnf, json_schema_to_grammar_exprs, JsonSchemaConverter};
use sep1::json_serialization::{JSONConvertible, JSONNode};
use sep1::precompute4::template_nwa::build_template_dwas;
use sep1::precompute4::characterize::compute_all_characterizations;
use sep1::tokenizer::LLMTokenID;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;

type RustGSS = LeveledGSS<ParseStateEdgeContent, Acc>;

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
    fn eat_u8_seq(s: &[u8]) -> Self {
        Self {
            inner: eat_u8_seq(s.to_vec()),
        }
    }

    #[staticmethod]
    pub fn eat_u8_set(set: Vec<u8>) -> Self {
        let u8set = U8Set::from_bytes(&set);
        Self {
            inner: eat_u8_set(u8set),
        }
    }

    #[staticmethod]
    fn eat_u8_negation(c: u8) -> Self {
        Self {
            inner: eat_u8_negation(c),
        }
    }

    #[staticmethod]
    fn eat_u8_set_negation(set: Vec<u8>) -> Self {
        let u8set = U8Set::from_bytes(&set);
        Self {
            inner: eat_u8_set_negation(u8set),
        }
    }

    #[staticmethod]
    pub fn eat_any() -> Self {
        Self {
            inner: eat_any_fast(),
        }
    }

    #[staticmethod]
    fn rep(expr: PyRegexExpr) -> Self {
        Self {
            inner: rep(expr.inner),
        }
    }

    #[staticmethod]
    fn rep1(expr: PyRegexExpr) -> Self {
        Self {
            inner: rep1(expr.inner),
        }
    }

    #[staticmethod]
    fn opt(expr: PyRegexExpr) -> Self {
        Self {
            inner: opt(expr.inner),
        }
    }

    #[staticmethod]
    fn prec(precedence: isize, expr: PyRegexExpr) -> PyRegexGroup {
        PyRegexGroup {
            inner: prec(precedence, expr.inner),
        }
    }

    #[staticmethod]
    fn eps() -> Self {
        Self { inner: eps() }
    }

    #[staticmethod]
    fn seq(exprs: Vec<PyRegexExpr>) -> Self {
        Self {
            inner: regex_seq(exprs.into_iter().map(|e| e.inner).collect()),
        }
    }

    #[staticmethod]
    fn choice(exprs: Vec<PyRegexExpr>) -> Self {
        Self {
            inner: regex_choice(exprs.into_iter().map(|e| e.inner).collect()),
        }
    }

    fn build(&self) -> PyResult<PyRegex> {
        Ok(PyRegex {
            inner: self.inner.clone().build(),
        })
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
        Self {
            inner: greedy_group(expr.inner),
        }
    }

    #[staticmethod]
    fn non_greedy_group(expr: PyRegexExpr) -> Self {
        Self {
            inner: non_greedy_group(expr.inner),
        }
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
        Ok(PyRegex {
            inner: self.inner.clone().build(),
        })
    }
}

#[pyclass(name = "Regex")]
#[derive(Clone)]
pub struct PyRegex {
    inner: Regex,
}

#[pymethods]
impl PyRegex {
    fn execute_from_state(
        &self,
        bytes: &[u8],
        state_id: usize,
    ) -> PyResult<(Option<usize>, Vec<(usize, usize)>)> {
        let exec_result = self
            .inner
            .execute_from_state(bytes, sep1::tokenizer::TokenizerStateID(state_id));
        let end_state = exec_result.end_state;
        let matches: Vec<(usize, usize)> =
            exec_result.matches.into_iter().map(|m| (m.id, m.width)).collect();
        Ok((end_state, matches))
    }

    fn tokens_accessible_from_state(&self, state_id: usize) -> PyResult<Vec<usize>> {
        let accessible = self
            .inner
            .tokens_accessible_from_state(sep1::tokenizer::TokenizerStateID(state_id));
        let out: Vec<usize> = accessible.into_iter().map(|tid| tid.0).collect();
        Ok(out)
    }

    fn initial_state_id(&self) -> usize {
        self.inner.initial_state_id().0
    }

    fn max_state(&self) -> usize {
        self.inner.max_state()
    }
}

#[pyclass(name = "GrammarDefinition")]
#[derive(Clone)]
pub struct PyGrammarDefinition {
    inner: GrammarDefinition,
}

#[pymethods]
impl PyGrammarDefinition {
    #[new]
    fn new(
        rules: Vec<(String, PyGrammarExpr)>,
        terminals: Vec<(String, PyRegexExpr)>,
    ) -> PyResult<Self> {
        let inner_rules = rules.into_iter().map(|(s, e)| (s, e.inner)).collect();
        let inner_terminals = terminals.into_iter().map(|(s, e)| (s, e.inner)).collect();

        let grammar_def = GrammarDefinition::from_exprs(inner_rules, inner_terminals).map_err(
            |e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to compile grammar: {}",
                    e
                ))
            },
        )?;

        Ok(PyGrammarDefinition { inner: grammar_def })
    }

    #[staticmethod]
    fn from_ebnf_file(path: &str) -> PyResult<Self> {
        let grammar_def = GrammarDefinition::from_ebnf_file(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to load or parse EBNF file '{}': {}",
                path, e
            ))
        })?;
        Ok(PyGrammarDefinition { inner: grammar_def })
    }

    /// Load an EBNF grammar from a file without optimization.
    /// Useful for visualization/debugging where you want to see the original grammar structure.
    #[staticmethod]
    fn from_ebnf_file_no_optimize(path: &str) -> PyResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to read EBNF file '{}': {}",
                path, e
            ))
        })?;
        let grammar_def = GrammarDefinition::from_ebnf_no_optimize(&content).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse EBNF file '{}': {}",
                path, e
            ))
        })?;
        Ok(PyGrammarDefinition { inner: grammar_def })
    }

    /// Create a GrammarDefinition from a Lark grammar string.
    /// 
    /// Lark format uses `:` for rule definitions and newlines as terminators:
    /// ```
    /// rule: expr
    /// ```
    #[staticmethod]
    fn from_lark(lark_source: &str) -> PyResult<Self> {
        let grammar_def = GrammarDefinition::from_lark(lark_source).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse Lark grammar: {}",
                e
            ))
        })?;
        Ok(PyGrammarDefinition { inner: grammar_def })
    }

    /// Create a GrammarDefinition from a Lark grammar file.
    #[staticmethod]
    fn from_lark_file(path: &str) -> PyResult<Self> {
        let grammar_def = GrammarDefinition::from_lark_file(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to load or parse Lark file '{}': {}",
                path, e
            ))
        })?;
        Ok(PyGrammarDefinition { inner: grammar_def })
    }

    fn optimize(&mut self) {
        self.inner.optimize();
    }

    fn compile(&self) -> PyResult<PyCompiledGrammar> {
        let compiled_grammar = CompiledGrammar::from_definition(Arc::new(self.inner.clone()));
        Ok(PyCompiledGrammar {
            inner: compiled_grammar,
        })
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
        let json_node = JSONNode::from_json_string(json_str).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse JSON string to JSONNode: {}",
                e
            ))
        })?;
        let grammar = GrammarDefinition::from_json(json_node).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to deserialize GrammarDefinition from JSONNode: {}",
                e
            ))
        })?;
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
        let json_node = JSONNode::from_json_string(json_str).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse JSON string to JSONNode: {}",
                e
            ))
        })?;
        let grammar = CompiledGrammar::from_json(json_node).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to deserialize CompiledGrammar from JSONNode: {}",
                e
            ))
        })?;
        Ok(PyCompiledGrammar { inner: grammar })
    }
}

#[pyclass(name = "GLRParser")]
#[derive(Clone)]
pub struct PyGLRParser {
    inner: Arc<GLRParser>,
}

#[pymethods]
impl PyGLRParser {
    fn get_parse_table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let table_dict = PyDict::new_bound(py);
        for (&state_id, row) in &self.inner.table {
            let row_dict = PyDict::new_bound(py);

            let actions_dict = PyDict::new_bound(py);
            for (&terminal_id, action) in &row.get_shifts_and_reduces_map() {
                let py_action = match action {
                    sep1::glr::table::Stage7ShiftsAndReducesLookaheadValue::Shift(to_state) => {
                        ("shift", to_state.0).to_object(py)
                    }
                    sep1::glr::table::Stage7ShiftsAndReducesLookaheadValue::Reduce {
                        nonterminal_id,
                        len,
                        production_ids,
                    } => {
                        let pids: Vec<usize> = production_ids.iter().map(|p| p.0).collect();
                        PyTuple::new_bound(
                            py,
                            &[
                                "reduce".to_object(py),
                                nonterminal_id.0.to_object(py),
                                len.to_object(py),
                                pids.to_object(py),
                            ],
                        )
                        .to_object(py)
                    }
                    sep1::glr::table::Stage7ShiftsAndReducesLookaheadValue::Split {
                        shift,
                        reduces,
                    } => {
                        let py_reduces = PyDict::new_bound(py);
                        for (len, nts) in reduces {
                            let py_nts = PyDict::new_bound(py);
                            for (nt, pids) in nts {
                                let pids_vec: Vec<usize> = pids.iter().map(|p| p.0).collect();
                                py_nts.set_item(nt.0, pids_vec)?;
                            }
                            py_reduces.set_item(len, py_nts)?;
                        }
                        PyTuple::new_bound(
                            py,
                            &[
                                "split".to_object(py),
                                shift.map(|s| s.0).to_object(py),
                                py_reduces.to_object(py),
                            ],
                        )
                        .to_object(py)
                    }
                };
                actions_dict.set_item(terminal_id.0, py_action)?;
            }
            row_dict.set_item("shifts_and_reduces", actions_dict)?;

            let gotos_dict = PyDict::new_bound(py);
            for (&nonterminal_id, goto) in &row.gotos {
                let py_goto = (goto.state_id.map(|s| s.0), goto.accept).to_object(py);
                gotos_dict.set_item(nonterminal_id.0, py_goto)?;
            }
            row_dict.set_item("gotos", gotos_dict)?;

            table_dict.set_item(state_id.0, row_dict)?;
        }

        let result_dict = PyDict::new_bound(py);
        result_dict.set_item("start_state_id", self.inner.start_state_id.0)?;
        result_dict.set_item("table", table_dict)?;

        Ok(result_dict)
    }

    #[getter]
    fn ignore_terminal_id(&self) -> Option<usize> {
        self.inner.ignore_terminal_id.map(|tid| tid.0)
    }
    
    /// Get all template DFAs (one per terminal).
    /// 
    /// Template DFAs encode the "below-bottom characterization" - what parser
    /// actions to take when we look below the stack. The edge labels are
    /// encoded parser state IDs (positive = push, negative = pop).
    /// 
    /// Returns a dict mapping terminal_id -> dict with DFA structure.
    fn get_template_dfas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = build_template_dwas(&self.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to build template DWAs: {:?}",
                e
            ))
        })?;
        
        let result_dict = PyDict::new_bound(py);
        for (terminal_id, dwa) in result {
            let dwa_dict = PyDict::new_bound(py);
            
            // Start state
            dwa_dict.set_item("start_state", dwa.body.start_state)?;
            
            // States
            let states_list = pyo3::types::PyList::empty_bound(py);
            for (state_id, state) in dwa.states.0.iter().enumerate() {
                let state_dict = PyDict::new_bound(py);
                state_dict.set_item("id", state_id)?;
                
                // Final weight
                if let Some(ref w) = state.final_weight {
                    state_dict.set_item("is_final", true)?;
                    state_dict.set_item("final_weight", format!("{}", w))?;
                } else {
                    state_dict.set_item("is_final", false)?;
                }
                
                // Transitions: symbol -> target_state
                let trans_dict = PyDict::new_bound(py);
                for (&symbol, &target) in &state.transitions {
                    // symbol is i32, encode as string showing +/- for push/pop
                    // DEFAULT_TRANSITION_SYMBOL is i32::MAX - 1
                    let symbol_str = if symbol == i32::MAX - 1 {
                        "DEFAULT".to_string()
                    } else if symbol >= 0 {
                        format!("+{}", symbol)  // push (positive state ID)
                    } else {
                        format!("{}", symbol)  // pop (negative state ID)
                    };
                    trans_dict.set_item(symbol_str, target)?;
                }
                state_dict.set_item("transitions", trans_dict)?;
                
                states_list.append(state_dict)?;
            }
            dwa_dict.set_item("states", states_list)?;
            
            result_dict.set_item(terminal_id.0, dwa_dict)?;
        }
        
        Ok(result_dict)
    }
    
    /// Get all below-bottom characterizations (one per terminal).
    /// 
    /// Characterizations describe parser behavior when looking below the stack:
    /// - initial_shifts: [(initial_state, shift_state), ...]
    /// - initial_reduces: [(initial_state, len, nonterminal), ...]
    /// - reduce_characterizations: {nt -> {reveal_and_rereduces, reveal_goto_shift_escapes}}
    fn get_characterizations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let all_chars = compute_all_characterizations(&self.inner);
        
        let result_dict = PyDict::new_bound(py);
        for (terminal_id, bb) in all_chars {
            let char_dict = PyDict::new_bound(py);
            
            // initial_shifts
            let shifts_list = pyo3::types::PyList::empty_bound(py);
            for &(initial, shift) in &bb.initial_shifts {
                let tuple = PyTuple::new_bound(py, &[initial.0, shift.0]);
                shifts_list.append(tuple)?;
            }
            char_dict.set_item("initial_shifts", shifts_list)?;
            
            // initial_reduces
            let reduces_list = pyo3::types::PyList::empty_bound(py);
            for &(initial, len, nt) in &bb.initial_reduces {
                let tuple = PyTuple::new_bound(py, &[initial.0 as usize, len, nt.0]);
                reduces_list.append(tuple)?;
            }
            char_dict.set_item("initial_reduces", reduces_list)?;
            
            // reduce_characterizations
            let rc_dict = PyDict::new_bound(py);
            for (nt, rc) in &bb.reduce_characterizations {
                let nt_dict = PyDict::new_bound(py);
                
                let rereduces_list = pyo3::types::PyList::empty_bound(py);
                for &(revealed, len, target_nt) in &rc.reveal_and_rereduces {
                    let tuple = PyTuple::new_bound(py, &[revealed.0 as usize, len, target_nt.0]);
                    rereduces_list.append(tuple)?;
                }
                nt_dict.set_item("reveal_and_rereduces", rereduces_list)?;
                
                let escapes_list = pyo3::types::PyList::empty_bound(py);
                for &(revealed, goto, shift) in &rc.reveal_goto_shift_escapes {
                    let tuple = PyTuple::new_bound(py, &[revealed.0 as usize, goto.0 as usize, shift.0 as usize]);
                    escapes_list.append(tuple)?;
                }
                nt_dict.set_item("reveal_goto_shift_escapes", escapes_list)?;
                
                rc_dict.set_item(nt.0, nt_dict)?;
            }
            char_dict.set_item("reduce_characterizations", rc_dict)?;
            
            // all_nts
            let nts_list: Vec<usize> = bb.all_nts.iter().map(|nt| nt.0).collect();
            char_dict.set_item("all_nts", nts_list)?;
            
            result_dict.set_item(terminal_id.0, char_dict)?;
        }
        
        Ok(result_dict)
    }
}

#[pyclass(name = "GrammarConstraint")]
#[derive(Clone)]
pub struct PyGrammarConstraint {
    inner: Arc<GrammarConstraint>,
}

#[pymethods]
impl PyGrammarConstraint {
    /// Create a new GrammarConstraint.
    /// 
    /// # Arguments
    /// * `grammar` - The compiled grammar
    /// * `token_to_id` - A dictionary mapping token bytes to token IDs
    /// * `max_llm_token_id` - (Optional) The maximum token ID in the vocabulary.
    ///   If not provided, this is inferred from the highest ID in token_to_id.
    #[new]
    #[pyo3(signature = (grammar, token_to_id, max_llm_token_id=None))]
    fn new(
        py: Python,
        grammar: PyCompiledGrammar,
        token_to_id: &Bound<'_, PyDict>,
        max_llm_token_id: Option<usize>,
    ) -> PyResult<Self> {
        let mut llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
        let mut highest_id = 0usize;
        for (key, value) in token_to_id.iter() {
            let token = key.extract::<&[u8]>()?;
            let id = value.extract::<usize>()?;
            highest_id = highest_id.max(id);
            llm_token_map.insert(token.to_vec(), LLMTokenID(id));
        }
        
        // Use provided max_llm_token_id, or infer from vocab
        let effective_max = max_llm_token_id.unwrap_or(highest_id);

        let constraint = GrammarConstraint::from_compiled_grammar(
            grammar.inner.clone(), // Clone the CompiledGrammar
            llm_token_map,
            effective_max,
        );

        Ok(Self {
            inner: Arc::new(constraint),
        })
    }

    fn print(&self) {
        println!("PyGrammarConstraint (details not implemented for print)");
    }

    fn print_parser(&self) {
        println!("{}", self.inner.parser);
    }

    fn to_json_string(&self) -> PyResult<String> {
        Ok(self.inner.to_json().to_json_string())
    }

    #[staticmethod]
    fn from_json_string(json_str: &str) -> PyResult<Self> {
        let json_node = JSONNode::from_json_string(json_str).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse JSON string to JSONNode: {}",
                e
            ))
        })?;
        let constraint = GrammarConstraint::from_json(json_node).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to deserialize GrammarConstraint from JSONNode: {}",
                e
            ))
        })?;
        Ok(Self {
            inner: Arc::new(constraint),
        })
    }

    fn get_id_to_token_map(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        // Use the new vocab_trie iterator for getting token bytes
        for (original_id, token_bytes) in self.inner.vocab_trie.iter() {
            dict.set_item(original_id, token_bytes)?;
        }
        Ok(dict.into())
    }

    fn tokenizer(&self) -> PyRegex {
        PyRegex {
            inner: self.inner.tokenizer.clone(),
        }
    }

    fn glr_parser(&self) -> PyGLRParser {
        PyGLRParser {
            inner: Arc::new(self.inner.parser.clone()),
        }
    }

    fn possible_matches(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (tokenizer_state_id, terminal_map) in &self.inner.possible_matches {
            let terminal_dict = PyDict::new_bound(py);
            for (terminal_id, llm_token_bv) in terminal_map {
                terminal_dict
                    .set_item(terminal_id.0, PyHybridBitset { inner: llm_token_bv.clone() })?;
            }
            dict.set_item(tokenizer_state_id.0, terminal_dict)?;
        }
        Ok(dict.into())
    }

    pub fn internal_bv_to_original_bitset(&self, internal_bv: &PyHybridBitset) -> PyResult<PyBitset> {
        let original_bv = self.inner.parser_dwa_vocab.internal_bv_to_original(&internal_bv.inner);
        Ok(PyBitset { inner: original_bv })
    }

    pub fn internal_bv_to_original_hybrid_bitset(&self, internal_bv: &PyHybridBitset) -> PyResult<PyBitset> {
        let original_bv = self.inner.parser_dwa_vocab.internal_bv_to_original(&internal_bv.inner);
        Ok(PyBitset { inner: original_bv })
    }

    pub fn internal_bv_to_original(&self, it: Bound<'_, PyIterator>) -> PyResult<PyBitset> {
        let original_bv = self.inner.parser_dwa_vocab.internal_bv_to_original(&RangeSet::from_iter(it.extract::<Vec<usize>>()?));
        Ok(PyBitset { inner: original_bv })
    }

    #[getter]
    fn vocab(&self) -> PyStageVocab {
        PyStageVocab {
            inner: Arc::new(self.inner.parser_dwa_vocab.clone()),
        }
    }
}

#[pyclass(name = "StageVocab")]
#[derive(Clone)]
pub struct PyStageVocab {
    inner: Arc<StageVocab>,
}

#[pymethods]
impl PyStageVocab {
    #[staticmethod]
    fn from_json_string(json_str: &str) -> PyResult<Self> {
        let json_node = JSONNode::from_json_string(json_str).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse JSON string to JSONNode: {}",
                e
            ))
        })?;
        let vocab = StageVocab::from_json(json_node).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to deserialize StageVocab from JSONNode: {}",
                e
            ))
        })?;
        Ok(Self {
            inner: Arc::new(vocab),
        })
    }

    pub fn internal_bv_to_original(&self, internal_bv: &PyHybridBitset) -> PyResult<PyBitset> {
        let original_bv = self.inner.internal_bv_to_original(&internal_bv.inner);
        Ok(PyBitset { inner: original_bv })
    }
}

#[pyclass(name = "HybridBitsetIterator")]
struct PyHybridBitsetIterator {
    inner: Mutex<Box<dyn Iterator<Item = usize> + Send>>,
}

#[pymethods]
impl PyHybridBitsetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<usize> {
        slf.inner.lock().unwrap().next()
    }
}

#[pyclass(name = "HybridBitset")]
#[derive(Clone)]
pub struct PyHybridBitset {
    inner: RustHybridBitset,
}

#[pymethods]
impl PyHybridBitset {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustHybridBitset::zeros(),
        }
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self {
            inner: RustHybridBitset::zeros(),
        }
    }

    #[staticmethod]
    fn ones(len: usize) -> Self {
        Self {
            inner: RustHybridBitset::ones(len),
        }
    }

    #[staticmethod]
    fn from_indices(indices: Vec<usize>) -> Self {
        Self {
            inner: RustHybridBitset::from_iter(indices),
        }
    }

    #[staticmethod]
    fn from_ranges(ranges: Vec<(usize, usize)>) -> Self {
        // Flatten to match HybridBitset JSON format: [start1, end1, start2, end2, ...]
        let mut flat = Vec::with_capacity(ranges.len() * 2);
        for (start, end) in ranges {
            flat.push(start.to_json());
            flat.push(end.to_json());
        }
        let inner = RustHybridBitset::from_json(sep1::json_serialization::JSONNode::Array(flat))
            .expect("Bitset::from_ranges JSON");
        Self { inner }
    }

    fn to_indices(&self) -> Vec<usize> {
        self.inner.iter_indices().collect()
    }

    fn to_ranges(&self) -> Vec<(usize, usize)> {
        self.inner.iter_ranges().collect()
    }

    fn __iter__(&self) -> PyHybridBitsetIterator {
        PyHybridBitsetIterator {
            inner: Mutex::new(Box::new(self.inner.clone().into_iter())),
        }
    }

    fn iter_ranges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        let ranges: Vec<(usize, usize)> = self.inner.iter_ranges().collect();
        pyo3::types::PyList::new_bound(py, &ranges).as_ref().iter()
    }

    fn contains(&self, idx: usize) -> bool {
        self.inner.contains(idx)
    }

    fn insert(&mut self, idx: usize) {
        let _ = self.inner.insert(idx);
    }

    fn remove(&mut self, idx: usize) {
        let _ = self.inner.remove(idx);
    }

    fn set(&mut self, idx: usize, value: bool) {
        self.inner.set(idx, value);
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn union(&self, other: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset {
            inner: &self.inner | &other.inner,
        }
    }

    fn intersection(&self, other: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset {
            inner: &self.inner & &other.inner,
        }
    }

    fn __ior__(&mut self, other: &PyHybridBitset) {
        self.inner |= &other.inner;
    }

    fn __iand__(&mut self, other: &PyHybridBitset) {
        self.inner &= &other.inner;
    }

    fn __isub__(&mut self, other: &PyHybridBitset) {
        self.inner -= &other.inner;
    }

    fn __ixor__(&mut self, other: &PyHybridBitset) {
        self.inner ^= &other.inner;
    }

    fn difference(&self, other: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset {
            inner: &self.inner - &other.inner,
        }
    }

    fn symmetric_difference(&self, other: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset {
            inner: &self.inner ^ &other.inner,
        }
    }

    fn is_subset(&self, other: &PyHybridBitset) -> bool {
        self.inner.is_subset(&other.inner)
    }

    fn is_superset(&self, other: &PyHybridBitset) -> bool {
        self.inner.is_superset(&other.inner)
    }

    fn is_disjoint(&self, other: &PyHybridBitset) -> bool {
        self.inner.is_disjoint(&other.inner)
    }

    fn to_json_string(&self) -> String {
        self.inner.to_json().to_json_string()
    }

    #[staticmethod]
    fn from_json_string(s: &str) -> PyResult<Self> {
        let node = sep1::json_serialization::JSONNode::from_json_string(s).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("parse json: {}", e))
        })?;
        let inner = RustHybridBitset::from_json(node).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bitset from json: {}", e))
        })?;
        Ok(Self { inner })
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("HybridBitset({:?})", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }
}

impl From<RustHybridBitset> for PyHybridBitset {
    fn from(inner: RustHybridBitset) -> Self {
        Self { inner }
    }
}
impl From<PyHybridBitset> for RustHybridBitset {
    fn from(p: PyHybridBitset) -> Self {
        p.inner
    }
}

#[pyclass(name = "BitsetIterator")]
struct PyBitsetIterator {
    inner: Mutex<Box<dyn Iterator<Item = usize> + Send>>,
}

#[pymethods]
impl PyBitsetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<usize> {
        slf.inner.lock().unwrap().next()
    }
}

#[pyclass(name = "Bitset")]
#[derive(Clone)]
pub struct PyBitset {
    inner: RustBitset,
}

#[pymethods]
impl PyBitset {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustBitset::zeros(),
        }
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self {
            inner: RustBitset::zeros(),
        }
    }

    #[staticmethod]
    fn ones(len: usize) -> Self {
        Self {
            inner: RustBitset::ones(len),
        }
    }

    #[staticmethod]
    fn from_indices(indices: Vec<usize>) -> Self {
        Self {
            inner: RustBitset::from_iter(indices),
        }
    }

    fn to_indices(&self) -> Vec<usize> {
        self.inner.iter_indices().collect()
    }

    fn __iter__(&self) -> PyBitsetIterator {
        let indices: Vec<usize> = self.inner.iter_indices().collect();
        PyBitsetIterator {
            inner: Mutex::new(Box::new(indices.into_iter())),
        }
    }

    fn contains(&self, idx: usize) -> bool {
        self.inner.contains(idx)
    }

    fn insert(&mut self, idx: usize) {
        let _ = self.inner.insert(idx);
    }

    fn remove(&mut self, idx: usize) {
        let _ = self.inner.remove(idx);
    }

    fn set(&mut self, idx: usize, value: bool) {
        if value {
            self.inner.insert(idx);
        } else {
            self.inner.remove(idx);
        }
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn union(&self, other: &PyBitset) -> PyBitset {
        PyBitset {
            inner: &self.inner | &other.inner,
        }
    }

    fn intersection(&self, other: &PyBitset) -> PyBitset {
        PyBitset {
            inner: &self.inner & &other.inner,
        }
    }

    fn __ior__(&mut self, other: &PyBitset) {
        self.inner |= &other.inner;
    }

    fn __iand__(&mut self, other: &PyBitset) {
        self.inner &= &other.inner;
    }

    fn __isub__(&mut self, other: &PyBitset) {
        self.inner -= &other.inner;
    }

    fn __ixor__(&mut self, other: &PyBitset) {
        self.inner ^= &other.inner;
    }

    fn difference(&self, other: &PyBitset) -> PyBitset {
        PyBitset {
            inner: &self.inner - &other.inner,
        }
    }

    fn symmetric_difference(&self, other: &PyBitset) -> PyBitset {
        PyBitset {
            inner: &self.inner ^ &other.inner,
        }
    }

    fn is_subset(&self, other: &PyBitset) -> bool {
        (&self.inner & &other.inner) == self.inner
    }

    fn is_superset(&self, other: &PyBitset) -> bool {
        (&self.inner | &other.inner) == self.inner
    }

    fn is_disjoint(&self, other: &PyBitset) -> bool {
        (&self.inner & &other.inner).is_empty()
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("Bitset({:?})", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }
}

#[pyclass(name = "GSSNode")]
#[derive(Clone)]
pub struct PyGSSNode {
    inner: RustGSS,
}

#[pymethods]
impl PyGSSNode {
    #[new]
    fn new() -> Self {
        PyGSSNode {
            inner: RustGSS::from_stacks(&[(vec![], RustAcc::new_fresh())]),
        }
    }

    fn is_alive(&self) -> bool {
        !self.inner.is_empty()
    }

    fn is_ok(&self) -> bool {
        !self.inner.is_empty()
    }

    fn allowed_llm_tokens(&self) -> PyHybridBitset {
        PyHybridBitset::from(
            self.inner.reduce_acc().map_or(RustHybridBitset::zeros(), |acc| acc.llm_tokens_union),
        )
    }

    fn disallowed_terminals(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        if let Some(acc) = self.inner.reduce_acc() {
            for (&tokenizer_state_id, bitset) in &acc.terminals_union {
                dict.set_item(
                    tokenizer_state_id,
                    PyHybridBitset { inner: bitset.clone() },
                )?;
            }
        }
        Ok(dict.into())
    }

    fn clone_node(&self) -> PyGSSNode {
        self.clone()
    }

    fn print_stats(&self) {
        let stats = self.inner.stats();
        println!("{:#?}", stats);
    }

    fn __str__(&self) -> String {
        self.inner.to_graph_string(false)
    }

    fn flatten<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Vec<usize>, (PyHybridBitset, PyObject))>> {
        let flattened = self.inner.to_stacks();
        flattened
            .into_iter()
            .map(|(path, acc)| {
                let path_ids: Vec<usize> = path.into_iter().map(|edge| edge.state_id.0).collect();
                let py_llm_tokens = PyHybridBitset {
                    inner: acc.llm_tokens_union,
                };
                let py_terminals_union = PyDict::new_bound(py);
                for (&sid, bv) in &acc.terminals_union {
                    py_terminals_union.set_item(sid, PyHybridBitset { inner: bv.clone() })?;
                }
                Ok((path_ids, (py_llm_tokens, py_terminals_union.to_object(py))))
            })
            .collect()
    }

    fn __hash__(&self) -> PyResult<isize> {
        Err(pyo3::exceptions::PyTypeError::new_err("GSSNode is not hashable"))
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => {
                let mut a_stacks = self.inner.to_stacks();
                let mut b_stacks = other.inner.to_stacks();
                a_stacks.sort();
                b_stacks.sort();
                Ok(a_stacks == b_stacks)
            }
            CompareOp::Ne => {
                let mut a_stacks = self.inner.to_stacks();
                let mut b_stacks = other.inner.to_stacks();
                a_stacks.sort();
                b_stacks.sort();
                Ok(a_stacks != b_stacks)
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Only == and != are supported for GSSNode",
            )),
        }
    }

    fn depth(&self) -> usize {
        self.inner.max_depth() as usize
    }

    fn predecessors(&self) -> Vec<(usize, PyGSSNode)> {
        let mut result = Vec::new();
        for (edge_content, preds_by_depth) in self.inner.predecessors() {
            for pred_vec in preds_by_depth.values() {
                for pred_gss in pred_vec {
                    result.push((
                        edge_content.state_id.0,
                        PyGSSNode { inner: pred_gss.clone() },
                    ));
                }
            }
        }
        result
    }
}

#[pyfunction]
fn gss_merge_many_with_depth(nodes: Vec<PyGSSNode>, depth: usize) -> PyGSSNode {
    let mut it = nodes.into_iter();
    if let Some(first) = it.next() {
        let mut merged = first.inner;
        for node in it {
            merged = merged.merge(&node.inner);
        }
        if depth > 0 {
            merged = merged.fuse(Some(depth as isize));
        }
        PyGSSNode { inner: merged }
    } else {
        PyGSSNode {
            inner: RustGSS::empty(),
        }
    }
}

#[pyfunction]
fn gss_allow_only_llm_tokens_and_prune(node: &mut PyGSSNode, bv: &PyHybridBitset) {
    node.inner = node.inner.apply_and_prune(|acc| {
        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union &= &bv.inner;
        if new_acc.llm_tokens_union.is_empty() {
            None
        } else {
            Some(new_acc)
        }
    });
}

#[pyfunction]
fn gss_reset_llm_tokens(node: &mut PyGSSNode) {
    node.inner = node.inner.apply(|acc| {
        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union = RustHybridBitset::max_ones();
        new_acc
    });
}

#[pyfunction]
fn gss_prune_disallowed_terminals(
    node: &mut PyGSSNode,
    terminals_map: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let mut rust_terminals_map = BTreeMap::new();
    for (k, v) in terminals_map.iter() {
        let tokenizer_state_id = sep1::tokenizer::TokenizerStateID(k.extract::<usize>()?);
        let terminal_bv = v.extract::<PyRef<PyHybridBitset>>()?.inner.clone();
        rust_terminals_map.insert(tokenizer_state_id, terminal_bv);
    }

    node.inner = node.inner.apply_and_prune(|acc| {
        for (sid, bv) in &rust_terminals_map {
            if let Some(disallowed) = acc.terminals_union.get(&sid.0) {
                if bv.intersects(disallowed) {
                    return None;
                }
            }
        }
        Some(acc.clone())
    });
    Ok(())
}

#[pyfunction]
fn gss_prune_llm_tokens_by_disallowed_terminals(
    node: &mut PyGSSNode,
    possible_matches: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let mut rust_possible_matches = BTreeMap::new();
    for (k, v) in possible_matches.iter() {
        let tokenizer_state_id = sep1::tokenizer::TokenizerStateID(k.extract::<usize>()?);
        let terminal_map_py = v.downcast::<PyDict>()?;
        let mut terminal_map = BTreeMap::new();
        for (term_k, term_v) in terminal_map_py.iter() {
            let terminal_id = sep1::glr::table::TerminalID(term_k.extract::<usize>()?);
            let llm_token_bv = term_v.extract::<PyRef<PyHybridBitset>>()?.inner.clone();
            terminal_map.insert(terminal_id, llm_token_bv);
        }
        rust_possible_matches.insert(tokenizer_state_id, terminal_map);
    }

    node.inner = node.inner.apply_and_prune(|acc| {
        if acc.terminals_union.is_empty() {
            return Some(acc.clone());
        }
        let mut forbidden_llm_tokens = RustHybridBitset::zeros();
        for (&tokenizer_state_id, disallowed_in_state) in &acc.terminals_union {
            if disallowed_in_state.is_empty() { continue; }
            if let Some(state_matches) = rust_possible_matches.get(&sep1::tokenizer::TokenizerStateID(tokenizer_state_id)) {
                for (terminal_id, llm_tokens) in state_matches {
                    if disallowed_in_state.contains(terminal_id.0) {
                        forbidden_llm_tokens |= llm_tokens;
                    }
                }
            }
        }

        if forbidden_llm_tokens.is_empty() {
            return Some(acc.clone());
        }

        let mut new_acc = acc.clone();
        new_acc.llm_tokens_union -= &forbidden_llm_tokens;

        if new_acc.llm_tokens_union.is_empty() {
            None
        } else {
            Some(new_acc)
        }
    });
    Ok(())
}

#[pyfunction]
fn gss_map_allowed_terminals_tokenizer_states(
    node: &mut PyGSSNode,
    state_map: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let mut rust_state_map = BTreeMap::new();
    for (k, v) in state_map.iter() {
        let from_state = sep1::tokenizer::TokenizerStateID(k.extract::<usize>()?);
        let to_state = sep1::tokenizer::TokenizerStateID(v.extract::<usize>()?);
        rust_state_map.insert(from_state, to_state);
    }

    node.inner = node.inner.apply(|acc| {
        let mut new_map = BTreeMap::new();
        for (old, new) in &rust_state_map {
            if let Some(bv) = acc.terminals_union.get(&old.0) {
                new_map
                    .entry(new.0)
                    .and_modify(|b: &mut RustHybridBitset| *b |= bv.clone())
                    .or_insert_with(|| bv.clone());
            }
        }
        let mut na = acc.clone();
        na.terminals_union = new_map;
        na
    });
    Ok(())
}

#[pyfunction]
fn gss_fuse_predecessors(node: &mut PyGSSNode, levels: usize) {
    node.inner = node.inner.fuse(Some(levels as isize));
}

#[pyfunction]
fn gss_popn_collect(node: &PyGSSNode, n: usize) -> Vec<(usize, PyGSSNode)> {
    let popped = node.inner.popn(n as isize);
    let mut out = Vec::new();
    for edge in popped.peek() {
        let iso_inner = popped.isolate(Some(edge.clone()));
        let num_paths = iso_inner.num_paths();
        if num_paths > 0 {
            let gss_node = PyGSSNode { inner: iso_inner };
            for _ in 0..num_paths {
                out.push((edge.state_id.0, gss_node.clone()));
            }
        }
    }
    out
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
                    let state = c.inner.init();
                    Ok::<_, PyErr>(state)
                },
            }
            .try_build()?,
        })
    }

    fn clone(&self) -> Self {
        let constraint = self.inner.borrow_constraint().clone();
        let gss_map: BTreeMap<sep1::tokenizer::TokenizerStateID, RustGSS> =
            self.inner.with_inner(|state| {
                state
                    .state
                    .iter()
                    .map(|(id, glr_state)| (*id, glr_state.stack.clone()))
                    .collect()
            });

        PyGrammarConstraintState {
            inner: PyGrammarConstraintStateWrapperTryBuilder {
                constraint,
                inner_builder: move |c: &PyGrammarConstraint| {
                    // TODO: This requires a method on GrammarConstraint to build a state from a map of GSSs.
                    // Assuming `state_from_gss_map` exists for this purpose.
                    let state = c.inner.state_from_gss_map(&gss_map);
                    Ok::<_, PyErr>(state)
                },
            }
            .try_build()
            .expect("Failed to clone PyGrammarConstraintState"),
        }
    }

    fn is_active(&self) -> bool {
        self.inner.with_inner(|state| state.is_active())
    }

    fn is_valid(&self) -> bool {
        self.inner.with_inner(|state| state.is_valid())
    }

    fn __str__(&self) -> String {
        self.inner.with_inner(|state| format!("{}", state))
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn get_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bitset = self.inner.with_inner(|state| state.get_mask());

        let bools: Vec<bool> = if bitset.is_empty() {
            vec![]
        } else {
            let max_val = bitset.iter_indices().max().unwrap(); // Safe due to is_empty check
            let mut bools = vec![false; max_val + 1];
            for i in bitset.iter_indices() {
                bools[i] = true;
            }
            bools
        };
        Ok(bools.into_pyarray_bound(py))
    }

    fn get_mask_bv(&self) -> PyResult<PyBitset> {
        let bitset = self.inner.with_inner(|state| state.get_mask());
        Ok(PyBitset { inner: bitset })
    }

    /// Fill a numpy int32 array with the token bitmask (llguidance-compatible format).
    /// 
    /// The array should have shape (vocab_size + 31) // 32.
    /// Bits are packed in little-endian order within each int32.
    /// This writes directly to the array without intermediate allocations.
    fn fill_next_token_bitmask(&self, mut bitmask: PyReadwriteArray1<i32>) -> PyResult<()> {
        let slice = bitmask.as_slice_mut()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Array must be contiguous: {:?}", e)))?;
        self.inner.with_inner(|state| state.fill_mask_i32(slice));
        Ok(())
    }

    /// Compute mask and fill bitmask via raw pointer (for maximum performance).
    /// 
    /// # Safety
    /// The caller must ensure the pointer is valid for at least `size_bytes` bytes.
    unsafe fn fill_next_token_bitmask_ptr(&self, ptr: usize, size_bytes: usize) -> PyResult<()> {
        let num_i32s = size_bytes / 4;
        self.inner.with_inner(|state| state.fill_mask_i32_ptr(ptr as *mut i32, num_i32s));
        Ok(())
    }

    /// Returns the required buffer size in i32 elements for the mask.
    fn mask_buffer_size_i32(&self) -> usize {
        self.inner.with_inner(|state| state.mask_buffer_size_i32())
    }

    fn commit(&mut self, llm_token_id: usize) -> PyResult<()> {
        self.inner.with_inner_mut(|state| {
            state.commit(LLMTokenID(llm_token_id))
        }).map_err(|e| PyValueError::new_err(e))
    }

    fn commit_bytes(&mut self, llm_token_bytes: &[u8]) {
        self.inner.with_inner_mut(|state| {
            state.commit_bytes(llm_token_bytes);
        });
    }

    fn print_stats(&self) {
        self.inner.with_inner(|state| state.print_gss_stats());
    }

    fn compute_commit_maps(
        &self,
        py: Python,
        llm_token_bytes: &[u8],
    ) -> PyResult<(PyObject, PyObject)> {
        let (state_map, terminals_map) =
            self.inner.with_inner(|s| s.compute_commit_maps(llm_token_bytes));

        let py_state_map = PyDict::new_bound(py);
        for (k, v) in state_map {
            py_state_map.set_item(k.0, v.0)?;
        }

        let py_terminals_map = PyDict::new_bound(py);
        for (k, v) in terminals_map {
            py_terminals_map.set_item(k.0, PyHybridBitset { inner: v })?;
        }

        Ok((py_state_map.into(), py_terminals_map.into()))
    }

    fn get_state_map(&self) -> PyResult<BTreeMap<usize, PyGSSNode>> {
        let mut out = BTreeMap::new();
        self.inner.with_inner(|state| {
            for (tokenizer_state_id, glr_state) in &state.state {
                out.insert(
                    tokenizer_state_id.0,
                    PyGSSNode {
                        inner: glr_state.stack.clone(),
                    },
                );
            }
        });
        Ok(out)
    }

    fn set_state_map(&mut self, new_state: BTreeMap<usize, PyGSSNode>) -> PyResult<()> {
        self.inner.with_inner_mut(|state| {
            let mut new_b_tree_map = BTreeMap::new();
            for (tokenizer_state_id, gss_node) in new_state {
                let glr_state = GLRParserState {
                    parser: &state.parent.parser,
                    stack: gss_node.inner.clone(),
                };
                new_b_tree_map
                    .insert(sep1::tokenizer::TokenizerStateID(tokenizer_state_id), glr_state);
            }
            state.state = new_b_tree_map;
        });
        Ok(())
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
            }
            .try_build()?,
        })
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.inner.with_parser_mut(|p| p.feed(bytes));
    }

    fn is_valid(&self) -> bool {
        self.inner.with_parser(|p| p.is_valid())
    }
}

/// Convert a JSON Schema (as JSON string) to EBNF grammar string.
#[pyfunction]
fn json_schema_to_ebnf_py(schema_json: &str) -> PyResult<String> {
    json_schema_to_ebnf(schema_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

/// Convert a JSON Schema (as JSON string) to a list of (name, GrammarExpr) pairs.
#[pyfunction]
fn json_schema_to_grammar_exprs_py(schema_json: &str) -> PyResult<Vec<(String, PyGrammarExpr)>> {
    let rules = json_schema_to_grammar_exprs(schema_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    Ok(rules.into_iter()
        .map(|(name, expr)| (name, PyGrammarExpr { inner: expr }))
        .collect())
}

/// Create a GrammarDefinition from a JSON Schema (as JSON string).
#[pyfunction]
fn grammar_definition_from_json_schema(schema_json: &str) -> PyResult<PyGrammarDefinition> {
    // First convert to EBNF, then parse it
    let ebnf = json_schema_to_ebnf(schema_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    let grammar_def = GrammarDefinition::from_ebnf(&ebnf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    Ok(PyGrammarDefinition { inner: grammar_def })
}

#[pymodule]
fn _sep1(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrammarExpr>()?;
    m.add_class::<PyRegexExpr>()?;
    m.add_class::<PyRegexGroup>()?;
    m.add_class::<PyRegexGroups>()?;
    m.add_class::<PyRegex>()?;
    m.add_class::<PyGrammarDefinition>()?;
    m.add_class::<PyCompiledGrammar>()?;
    m.add_class::<PyGLRParser>()?;
    m.add_class::<PyGrammarConstraint>()?;
    m.add_class::<PyStageVocab>()?;
    m.add_class::<PyGrammarConstraintState>()?;
    m.add_class::<PyHybridBitsetIterator>()?;
    m.add_class::<PyHybridBitset>()?;
    m.add_class::<PyBitsetIterator>()?;
    m.add_class::<PyBitset>()?;
    m.add_class::<PyGSSNode>()?;
    m.add_function(wrap_pyfunction!(gss_merge_many_with_depth, m)?)?;
    m.add_function(wrap_pyfunction!(gss_allow_only_llm_tokens_and_prune, m)?)?;
    m.add_function(wrap_pyfunction!(gss_reset_llm_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(gss_prune_disallowed_terminals, m)?)?;
    m.add_function(wrap_pyfunction!(
        gss_prune_llm_tokens_by_disallowed_terminals,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gss_map_allowed_terminals_tokenizer_states,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(gss_fuse_predecessors, m)?)?;
    m.add_function(wrap_pyfunction!(gss_popn_collect, m)?)?;
    // JSON Schema conversion functions
    m.add_function(wrap_pyfunction!(json_schema_to_ebnf_py, m)?)?;
    m.add_function(wrap_pyfunction!(json_schema_to_grammar_exprs_py, m)?)?;
    m.add_function(wrap_pyfunction!(grammar_definition_from_json_schema, m)?)?;
    // Benchmark functions
    m.add_function(wrap_pyfunction!(set_benchmark_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_last_mask_time_ns, m)?)?;
    m.add_class::<PyIncrementalParser>()?;
    Ok(())
}

/// Enable benchmark mode to capture Rust-native timings.
/// When enabled, fill_next_token_bitmask will record its execution time
/// which can be retrieved via get_last_mask_time_ns().
#[pyfunction]
fn set_benchmark_mode(enabled: bool) {
    sep1::constraint_fns::set_benchmark_mode(enabled);
}

/// Get the last mask computation time in nanoseconds.
/// Only valid if benchmark mode is enabled.
#[pyfunction]
fn get_last_mask_time_ns() -> u64 {
    sep1::constraint_fns::get_last_mask_time_ns()
}