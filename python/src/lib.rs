use sep1::tokenizer::LLMTokenID;
use sep1::finite_automata::{Expr as RegexExpr, ExprGroups as RegexGroups, greedy_group, non_greedy_group, groups as regex_groups, _choice as regex_choice, eat_u8, eat_u8_negation, eat_u8_set, eps, opt, prec, rep, rep1, _seq as regex_seq, ExprGroups, eat_u8_seq, eat_u8_set_negation};
use sep1::finite_automata::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use sep1::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use sep1::glr::parser::{GLRParser, GLRParserState};
use sep1::glr::table::{generate_glr_parser, StateID, TerminalID};
use sep1::interface::{CompiledGrammar, GrammarExpr, choice as grammar_choice, literal as grammar_literal, optional as grammar_optional, repeat as grammar_repeat, r#ref as grammar_ref, sequence as grammar_sequence, eat_any_fast, GrammarDefinition};
use sep1::constraint::{GrammarConstraint, GrammarConstraintState};
use std::collections::{BTreeMap, BTreeSet};
use bimap::BiBTreeMap;
use pyo3::basic::CompareOp;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use ouroboros::self_referencing;
use numpy::{IntoPyArray, PyArray1};
use sep1::datastructures::u8set::U8Set;
use sep1::interface::IncrementalParser;
use sep1::json_serialization::{JSONConvertible, JSONNode};
use sep1::datastructures::hybrid_bitset::HybridBitset as RustHybridBitset;
use sep1::datastructures::gss::{GSSNode as RustGSSNode, allow_only_llm_tokens_and_prune as rust_allow_only, popn_collect_isolated_parents as rust_popn_collect, GSSNode, gather_gss_stats, popn_collect_fast};

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
        Self { inner: eat_u8_seq(s.to_vec()) }
    }

    #[staticmethod]
    pub fn eat_u8_set(set: Vec<u8>) -> Self {
        let u8set = U8Set::from_bytes(&set);
        Self { inner: eat_u8_set(u8set) }
    }

    #[staticmethod]
    fn eat_u8_negation(c: u8) -> Self {
        Self { inner: eat_u8_negation(c) }
    }

    #[staticmethod]
    fn eat_u8_set_negation(set: Vec<u8>) -> Self {
        let u8set = U8Set::from_bytes(&set);
        Self { inner: eat_u8_set_negation(u8set) }
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
    fn execute_from_state(&self, bytes: &[u8], state_id: usize) -> PyResult<(Option<usize>, Vec<(usize, usize)>)> {
        let exec_result = self.inner.execute_from_state(bytes, sep1::tokenizer::TokenizerStateID(state_id));
        let end_state = exec_result.end_state;
        let matches: Vec<(usize, usize)> = exec_result.matches.into_iter().map(|m| (m.id, m.width)).collect();
        Ok((end_state, matches))
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
    fn new(rules: Vec<(String, PyGrammarExpr)>, terminals: Vec<(String, PyRegexExpr)>) -> PyResult<Self> {
        let inner_rules = rules.into_iter().map(|(s, e)| (s, e.inner)).collect();
        let inner_terminals = terminals.into_iter().map(|(s, e)| (s, e.inner)).collect();

        let grammar_def = GrammarDefinition::from_exprs(inner_rules, inner_terminals)
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to compile grammar: {}", e)))?;

        Ok(PyGrammarDefinition { inner: grammar_def })
    }

    #[staticmethod]
    fn from_ebnf_file(path: &str) -> PyResult<Self> {
        let grammar_def = GrammarDefinition::from_ebnf_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to load or parse EBNF file '{}': {}", path, e)))?;
        Ok(PyGrammarDefinition { inner: grammar_def })
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
            for (&terminal_id, action) in &row.shifts_and_reduces_full {
                let py_action = match action {
                    sep1::glr::table::Stage7ShiftsAndReducesLookaheadValue::Shift(to_state) => {
                        PyTuple::new_bound(py, &["shift", to_state.0]).to_object(py)
                    }
                    sep1::glr::table::Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                        let pids: Vec<usize> = production_ids.iter().map(|p| p.0).collect();
                        PyTuple::new_bound(py, &[
                            "reduce".to_object(py),
                            nonterminal_id.0.to_object(py),
                            len.to_object(py),
                            pids.to_object(py)
                        ]).to_object(py)
                    }
                    sep1::glr::table::Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                        let py_reduces = PyDict::new_bound(py);
                        for (len, nts) in reduces {
                            let py_nts = PyDict::new_bound(py);
                            for (nt, pids) in nts {
                                let pids_vec: Vec<usize> = pids.iter().map(|p| p.0).collect();
                                py_nts.set_item(nt.0, pids_vec)?;
                            }
                            py_reduces.set_item(len, py_nts)?;
                        }
                        PyTuple::new_bound(py, &[
                            "split".to_object(py),
                            shift.map(|s| s.0).to_object(py),
                            py_reduces.to_object(py)
                        ]).to_object(py)
                    }
                };
                actions_dict.set_item(terminal_id.0, py_action)?;
            }
            row_dict.set_item("shifts_and_reduces", actions_dict)?;

            let gotos_dict = PyDict::new_bound(py);
            for (&nonterminal_id, goto) in &row.gotos {
                let py_goto = PyTuple::new_bound(py, &[goto.state_id.map(|s| s.0), goto.accept]);
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
}

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
            LLMTokenID(0),
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

    fn precompute2_json_string(&self) -> PyResult<String> {
        // Build [roots_map, arena] JSON array
        let roots_arr = sep1::json_serialization::JSONNode::Array(
            self.inner.precomputed2.iter()
                .map(|(sid, idx)| sep1::json_serialization::JSONNode::Array(vec![
                    sid.0.to_json(), idx.to_json()
                ]))
                .collect()
        );
        let arena_json = self.inner.trie2_god.to_json();
        let top = sep1::json_serialization::JSONNode::Array(vec![roots_arr, arena_json]);
        Ok(top.to_json_string())
    }

    fn precompute3_json_string(&self) -> PyResult<String> {
        let roots_arr = sep1::json_serialization::JSONNode::Array(
            self.inner.precomputed3.iter()
                .map(|(sid, idx)| sep1::json_serialization::JSONNode::Array(vec![
                    sid.0.to_json(), idx.to_json()
                ]))
                .collect()
        );
        let arena_json = self.inner.trie3_god.to_json();
        let top = sep1::json_serialization::JSONNode::Array(vec![roots_arr, arena_json]);
        Ok(top.to_json_string())
    }

    fn get_id_to_token_map(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (token_bytes, token_id) in self.inner.llm_vocab.llm_token_map.iter() {
            dict.set_item(token_id.0, token_bytes.as_slice())?;
        }
        Ok(dict.into())
    }

    fn original_to_internal_map(&self) -> PyResult<std::collections::BTreeMap<usize, usize>> {
        let mut m = std::collections::BTreeMap::new();
        for (orig, intl) in self.inner.llm_vocab.original_to_internal_id_bimap.iter() {
            m.insert(*orig, *intl);
        }
        Ok(m)
    }

    fn internal_to_original_map(&self) -> PyResult<std::collections::BTreeMap<usize, usize>> {
        let mut m = std::collections::BTreeMap::new();
        for (orig, intl) in self.inner.llm_vocab.original_to_internal_id_bimap.iter() {
            m.insert(*intl, *orig);
        }
        Ok(m)
    }

    fn internal_bv_to_original(&self, bv: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset::from(self.inner.internal_bv_to_original(&bv.inner))
    }

    fn original_bv_to_internal(&self, bv: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset::from(self.inner.original_bv_to_internal(&bv.inner))
    }

    fn tokenizer(&self) -> PyRegex {
        PyRegex { inner: self.inner.tokenizer.clone() }
    }

    fn glr_parser(&self) -> PyGLRParser {
        PyGLRParser { inner: Arc::new(self.inner.parser.clone()) }
    }

    fn possible_matches(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (tokenizer_state_id, terminal_map) in &self.inner.possible_matches {
            let terminal_dict = PyDict::new_bound(py);
            for (terminal_id, llm_token_bv) in terminal_map {
                terminal_dict.set_item(terminal_id.0, PyHybridBitset { inner: llm_token_bv.clone() })?;
            }
            dict.set_item(tokenizer_state_id.0, terminal_dict)?;
        }
        Ok(dict.into())
    }
}

#[pyclass(name = "Bitset")]
#[derive(Clone)]
pub struct PyHybridBitset {
    inner: RustHybridBitset,
}

#[pymethods]
impl PyHybridBitset {
    #[new]
    fn new() -> Self {
        Self { inner: RustHybridBitset::zeros() }
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self { inner: RustHybridBitset::zeros() }
    }

    #[staticmethod]
    fn ones(len: usize) -> Self {
        Self { inner: RustHybridBitset::ones(len) }
    }

    #[staticmethod]
    fn from_indices(indices: Vec<usize>) -> Self {
        Self { inner: RustHybridBitset::from_iter(indices) }
    }

    #[staticmethod]
    fn from_ranges(ranges: Vec<(usize, usize)>) -> Self {
        let json_ranges: Vec<Vec<usize>> = ranges.into_iter().map(|(s,e)| vec![s,e]).collect();
        let inner = RustHybridBitset::from_json(sep1::json_serialization::JSONNode::Array(
            json_ranges.into_iter().map(|p| sep1::json_serialization::JSONNode::Array(vec![p[0].to_json(), p[1].to_json()])).collect()
        )).expect("Bitset::from_ranges JSON");
        Self { inner }
    }

    fn to_indices(&self) -> Vec<usize> {
        self.inner.iter().collect()
    }

    fn to_ranges(&self) -> Vec<(usize, usize)> {
        // reuse JSON conversion for simplicity
        let json = self.inner.to_json();
        let arr = match json {
            sep1::json_serialization::JSONNode::Array(arr) => arr,
            _ => vec![],
        };
        let mut out = Vec::new();
        for pair in arr {
            if let sep1::json_serialization::JSONNode::Array(v) = pair {
                if v.len() == 2 {
                    let s = usize::from_json(v[0].clone()).unwrap();
                    let e = usize::from_json(v[1].clone()).unwrap();
                    out.push((s,e));
                }
            }
        }
        out
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
        PyHybridBitset { inner: &self.inner | &other.inner }
    }

    fn intersection(&self, other: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset { inner: &self.inner & &other.inner }
    }

    fn difference(&self, other: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset { inner: &self.inner - &other.inner }
    }

    fn symmetric_difference(&self, other: &PyHybridBitset) -> PyHybridBitset {
        PyHybridBitset { inner: &self.inner ^ &other.inner }
    }

    fn to_json_string(&self) -> String {
        self.inner.to_json().to_json_string()
    }

    #[staticmethod]
    fn from_json_string(s: &str) -> PyResult<Self> {
        let node = sep1::json_serialization::JSONNode::from_json_string(s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError,_>(format!("parse json: {}", e)))?;
        let inner = RustHybridBitset::from_json(node)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError,_>(format!("bitset from json: {}", e)))?;
        Ok(Self { inner })
    }
}

impl From<RustHybridBitset> for PyHybridBitset {
    fn from(inner: RustHybridBitset) -> Self { Self { inner } }
}
impl From<PyHybridBitset> for RustHybridBitset {
    fn from(p: PyHybridBitset) -> Self { p.inner }
}

#[pyclass(name = "GSSNode")]
#[derive(Clone)]
pub struct PyGSSNode {
    inner: std::sync::Arc<RustGSSNode>,
}

#[pymethods]
impl PyGSSNode {
    #[new]
    fn new() -> Self {
        PyGSSNode { inner: std::sync::Arc::new(RustGSSNode::new_fresh()) }
    }

    fn ptr(&self) -> usize {
        // Stable identity for the inner Arc<GSSNode>; useful for change detection
        let raw = std::sync::Arc::as_ptr(&self.inner);
        raw as usize
    }

    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    fn is_ok(&self) -> bool {
        self.inner.is_ok()
    }

    fn allowed_llm_tokens(&self) -> PyHybridBitset {
        PyHybridBitset::from(self.inner.allowed_llm_tokens())
    }

    fn clone_node(&self) -> PyGSSNode {
        self.clone()
    }

    fn popn_fast(&self, n: usize) -> Vec<(usize, PyGSSNode)> {
        let pairs = popn_collect_fast(&self.inner, n);
        pairs.into_iter()
            .map(|(sid, arc)| (sid.0, PyGSSNode { inner: arc }))
            .collect()
    }

    fn print_stats(&self) {
        let stats = gather_gss_stats(&[self.inner.as_ref()]);
        println!("{:#?}", stats);
    }

    fn __hash__(&self) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(*self.inner == *other.inner),
            CompareOp::Ne => Ok(*self.inner != *other.inner),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err("Only == and != are supported for GSSNode")),
        }
    }
}

#[pyfunction]
fn gss_merge_many_with_depth(nodes: Vec<PyGSSNode>, depth: usize) -> PyGSSNode {
    let arcs = nodes.into_iter().map(|n| n.inner.clone()).collect::<Vec<_>>();
    let merged = RustGSSNode::merge_many_with_depth(depth, arcs);
    PyGSSNode { inner: merged }
}

#[pyfunction]
fn gss_allow_only_llm_tokens_and_prune(node: &mut PyGSSNode, bv: &PyHybridBitset) {
    let mut arc = node.inner.clone();
    rust_allow_only(&mut arc, &bv.inner);
    node.inner = arc;
}

#[pyfunction]
fn gss_reset_llm_tokens(node: &mut PyGSSNode) {
    let mut arc = node.inner.clone();
    sep1::datastructures::gss::reset_llm_tokens(&mut arc, &mut std::collections::HashMap::new());
    node.inner = arc;
}

#[pyfunction]
fn gss_prune_disallowed_terminals(node: &mut PyGSSNode, terminals_map: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut rust_terminals_map = BTreeMap::new();
    for (k, v) in terminals_map.iter() {
        let tokenizer_state_id = sep1::tokenizer::TokenizerStateID(k.extract::<usize>()?);
        let terminal_bv = v.extract::<PyRef<PyHybridBitset>>()?.inner.clone();
        rust_terminals_map.insert(tokenizer_state_id, terminal_bv);
    }
    
    let mut arc = node.inner.clone();
    sep1::datastructures::gss::prune_disallowed_terminals(&mut arc, &rust_terminals_map, &mut std::collections::HashMap::new());
    node.inner = arc;
    Ok(())
}

#[pyfunction]
fn gss_map_allowed_terminals_tokenizer_states(node: &mut PyGSSNode, state_map: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut rust_state_map = BTreeMap::new();
    for (k, v) in state_map.iter() {
        let from_state = sep1::tokenizer::TokenizerStateID(k.extract::<usize>()?);
        let to_state = sep1::tokenizer::TokenizerStateID(v.extract::<usize>()?);
        rust_state_map.insert(from_state, to_state);
    }

    let mut arc = node.inner.clone();
    sep1::datastructures::gss::map_allowed_terminals_tokenizer_states(&mut arc, &rust_state_map, &mut std::collections::HashMap::new());
    node.inner = arc;
    Ok(())
}

#[pyfunction]
fn gss_fuse_predecessors(node: &mut PyGSSNode, levels: usize) {
    let mut arc = node.inner.clone();
    arc = sep1::datastructures::gss::fuse_predecessors_recursive(&arc, levels, &mut std::collections::HashMap::new());
    node.inner = arc;
}

#[pyfunction]
fn gss_popn_collect(node: &PyGSSNode, n: usize) -> Vec<(usize, PyGSSNode)> {
    let pairs = rust_popn_collect(&node.inner, n);
    pairs.into_iter()
        .map(|(sid, arc)| (sid.0, PyGSSNode { inner: arc }))
        .collect()
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
            .try_build()?
        })
    }

    fn get_mask<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bitset = self.inner.with_inner_mut(|state| state.get_mask());
        let bools: Vec<bool> = bitset.iter_bits().collect();
        Ok(bools.into_pyarray_bound(py))
    }

    fn get_mask_bv(&mut self) -> PyResult<PyHybridBitset> {
        let bitset = self.inner.with_inner_mut(|state| state.get_mask());
        Ok(PyHybridBitset { inner: bitset })
    }

    fn commit(&mut self, llm_token_id: usize) {
        // println!("Committing token {} to grammar constraint state", llm_token_id); // Debug
        self.inner.with_inner_mut(|state| {
            state.commit(LLMTokenID(llm_token_id));
        });
    }

    fn print_stats(&self) {
        self.inner.with_inner(|state| state.print_gss_stats());
    }

    fn get_state_map(&self) -> PyResult<BTreeMap<usize, PyGSSNode>> {
        let mut out = BTreeMap::new();
        self.inner.with_inner(|state| {
            for (tokenizer_state_id, glr_state) in &state.state {
                out.insert(tokenizer_state_id.0, PyGSSNode { inner: glr_state.active_state.stack.clone() });
            }
        });
        Ok(out)
    }

    fn set_state_map(&mut self, new_state: BTreeMap<usize, PyGSSNode>) -> PyResult<()> {
        self.inner.with_inner_mut(|state| {
            let mut new_b_tree_map = BTreeMap::new();
            for (tokenizer_state_id, gss_node) in new_state {
                let glr_state = state.parent.parser.init_glr_parser_from_stack(gss_node.inner.clone());
                new_b_tree_map.insert(sep1::tokenizer::TokenizerStateID(tokenizer_state_id), glr_state);
            }
            state.state = new_b_tree_map;
        });
        Ok(())
    }

    fn filtered_state_gss_map(&self) -> PyResult<std::collections::BTreeMap<usize, PyGSSNode>> {
        let mut out = std::collections::BTreeMap::new();
        self.inner.with_inner(|state| {
            for (tokenizer_state_id, glr_state) in &state.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            let disallowed_l2 = glr_state.active_state.stack.disallowed_terminals();
            let mut forbidden = RustHybridBitset::zeros();

            for (range, disallowed_terminals_for_range) in disallowed_l2.range_values() {
                if disallowed_terminals_for_range.is_empty() { continue; }
                let possible_matches = &state.parent.possible_matches;
                let slice = possible_matches.range(sep1::tokenizer::TokenizerStateID(*range.start())..=sep1::tokenizer::TokenizerStateID(*range.end()));
                for (_sid, per_state) in slice {
                    for (terminal_id, llm_bv) in per_state {
                        if disallowed_terminals_for_range.contains(terminal_id.0) {
                            forbidden |= llm_bv.clone();
                        }
                    }
                }
                }

            let mut gss_arc = glr_state.active_state.stack.clone();
            if !forbidden.is_empty() {
                let allowed = &RustHybridBitset::max_ones() - &forbidden;
                rust_allow_only(&mut gss_arc, &allowed);
            }
            out.insert(tokenizer_state_id.0, PyGSSNode { inner: gss_arc });
        }
        });
        Ok(out)
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
    m.add_class::<PyGrammarDefinition>()?;
    m.add_class::<PyCompiledGrammar>()?;
    m.add_class::<PyGLRParser>()?;
    m.add_class::<PyGrammarConstraint>()?;
    m.add_class::<PyGrammarConstraintState>()?;
    m.add_class::<PyHybridBitset>()?;
    m.add_class::<PyGSSNode>()?;
    m.add_function(wrap_pyfunction!(gss_merge_many_with_depth, m)?)?;
    m.add_function(wrap_pyfunction!(gss_allow_only_llm_tokens_and_prune, m)?)?;
    m.add_function(wrap_pyfunction!(gss_reset_llm_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(gss_prune_disallowed_terminals, m)?)?;
    m.add_function(wrap_pyfunction!(gss_map_allowed_terminals_tokenizer_states, m)?)?;
    m.add_function(wrap_pyfunction!(gss_fuse_predecessors, m)?)?;
    m.add_function(wrap_pyfunction!(gss_popn_collect, m)?)?;
    m.add_class::<PyIncrementalParser>()?;
    Ok(())
}