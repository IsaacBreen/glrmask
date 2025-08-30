use std::collections::HashMap;
use sep1::tokenizer::LLMTokenID;
use sep1::finite_automata::{Expr as RegexExpr, ExprGroups as RegexGroups, greedy_group, non_greedy_group, groups as regex_groups, _choice as regex_choice, eat_u8, eat_u8_negation, eat_u8_set, eps, opt, prec, rep, rep1, _seq as regex_seq, ExprGroups, eat_u8_seq, eat_u8_set_negation};
use sep1::finite_automata::Regex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyList};
use sep1::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use sep1::glr::parser::{GLRParser, GLRParserState};
use sep1::glr::table::{generate_glr_parser, StateID, TerminalID};
use sep1::interface::{CompiledGrammar, GrammarExpr, choice as grammar_choice, literal as grammar_literal, optional as grammar_optional, repeat as grammar_repeat, r#ref as grammar_ref, sequence as grammar_sequence, eat_any_fast, GrammarDefinition};
use sep1::constraint::{GrammarConstraint, GrammarConstraintState};
use std::collections::{BTreeMap, BTreeSet};
use bimap::BiBTreeMap;
use std::sync::Arc;
use ouroboros::self_referencing;
use numpy::{IntoPyArray, PyArray1, ToPyArray};
use sep1::datastructures::gss::{GSSNode, GSSPopper, GSSPopperItem, GSSPopperItemPeek, allow_only_llm_tokens_and_prune_arc, disallow_llm_tokens_and_prune_arc};
use sep1::datastructures::hybrid_bitset::HybridBitset;
use sep1::datastructures::u8set::U8Set;
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

    fn precomputed3_to_json_string(&self) -> PyResult<String> {
        let mut obj = BTreeMap::new();
        obj.insert("precomputed3".to_string(), self.inner.precomputed3.to_json());
        obj.insert("trie3_god".to_string(), self.inner.trie3_god.to_json());
        Ok(JSONNode::Object(obj).to_json_string())
    }

    fn original_to_internal_llm_token_map(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for (original, internal) in self.inner.llm_vocab.original_to_internal_id_bimap.iter() {
            dict.set_item(original, internal)?;
        }
        Ok(dict.into())
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
        });
    }

    fn get_initial_trie3_states(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let state = self.inner.borrow_inner();

        for (tokenizer_state_id, glr_state) in &state.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_idx) = state.parent.precomputed3.get(tokenizer_state_id) {
                let mut forbidden_llm_tokens = HybridBitset::zeros();
                let disallowed_terminals_l2 = glr_state.active_state.stack.disallowed_terminals();

                for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
                    if disallowed_terminals_for_range.is_empty() {
                        continue;
                    }

                    let relevant_possible_matches = state.parent.possible_matches.range(
                        sep1::tokenizer::TokenizerStateID(*tokenizer_state_range.start())..=sep1::tokenizer::TokenizerStateID(*tokenizer_state_range.end())
                    );

                    for (_tokenizer_state_id, possible_matches_for_state) in relevant_possible_matches {
                        for (terminal_id, llm_tokens_that_match_this_terminal) in possible_matches_for_state {
                            if disallowed_terminals_for_range.contains(terminal_id.0) {
                                forbidden_llm_tokens |= llm_tokens_that_match_this_terminal;
                            }
                        }
                    }
                }

                let mut gss_node = glr_state.active_state.stack.clone();
                if !forbidden_llm_tokens.is_empty() {
                    disallow_llm_tokens_and_prune_arc(&mut gss_node, &forbidden_llm_tokens, &mut HashMap::new());
                }

                let py_gss_node = PyGSSNode { inner: gss_node };
                let trie_idx = precomputed_trie_root_idx.as_usize();

                if let Some(existing_node_bound) = dict.get_item(trie_idx)? {
                    let existing_node: PyGSSNode = existing_node_bound.extract()?;
                    let merged_node = PyGSSNode { inner: GSSNode::merge_many_with_depth(1, vec![existing_node.inner, py_gss_node.inner]) };
                    dict.set_item(trie_idx, merged_node.into_py(py))?;
                } else {
                    dict.set_item(trie_idx, py_gss_node.into_py(py))?;
                }
            }
        }
        Ok(dict.into())
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

#[pyclass(name = "HybridBitset")]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PyHybridBitset {
    inner: HybridBitset,
}

#[pymethods]
impl PyHybridBitset {
    #[new]
    fn new() -> Self {
        Self { inner: HybridBitset::zeros() }
    }

    #[staticmethod]
    fn zeros() -> Self {
        Self { inner: HybridBitset::zeros() }
    }

    #[staticmethod]
    fn from_ranges(ranges: Vec<(usize, usize)>) -> Self {
        use std::iter::FromIterator;
        let range_iter = ranges.into_iter().map(|(start, end)| start..=end);
        Self { inner: HybridBitset::from_iter(range_iter) }
    }

    fn contains(&self, index: usize) -> bool {
        self.inner.contains(index)
    }

    fn __and__(&self, other: &PyHybridBitset) -> Self {
        Self { inner: &self.inner & &other.inner }
    }

    fn __or__(&self, other: &PyHybridBitset) -> Self {
        Self { inner: &self.inner | &other.inner }
    }

    fn __ior__(&mut self, other: &PyHybridBitset) {
        self.inner |= &other.inner;
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn iter_bits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bools: Vec<bool> = self.inner.iter_bits().collect();
        Ok(bools.into_pyarray_bound(py))
    }
}

#[pyclass(name = "GSSNode")]
#[derive(Clone)]
pub struct PyGSSNode {
    inner: Arc<GSSNode>,
}

#[pymethods]
impl PyGSSNode {
    fn allow_only_llm_tokens_and_prune(&mut self, allowed_tokens: &PyHybridBitset) {
        allow_only_llm_tokens_and_prune_arc(&mut self.inner, &allowed_tokens.inner, &mut HashMap::new());
    }

    fn popn(&self, n: usize) -> PyGSSPopper {
        PyGSSPopper { inner: self.inner.popn(n) }
    }

    #[staticmethod]
    fn merge_many(nodes: &Bound<'_, PyList>, depth: usize) -> PyResult<Self> {
        let mut inner_nodes = Vec::new();
        for node_any in nodes.iter() {
            let node: PyRef<'_, PyGSSNode> = node_any.extract()?;
            inner_nodes.push(node.inner.clone());
        }
        Ok(Self { inner: GSSNode::merge_many_with_depth(depth, inner_nodes) })
    }

    fn is_ok(&self) -> bool {
        !self.inner.is_empty()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn allowed_llm_tokens(&self) -> PyHybridBitset {
        PyHybridBitset { inner: self.inner.allowed_llm_tokens() }
    }

    fn clone(&self) -> Self {
        self.clone()
    }
}

#[pyclass(name = "GSSPopper")]
pub struct PyGSSPopper {
    inner: GSSPopper,
}

#[pymethods]
impl PyGSSPopper {
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyGSSPopper>> {
        Ok(slf.into())
    }

    fn __next__(&mut self) -> Option<PyGSSPopperItem> {
        self.inner.next().map(|item| PyGSSPopperItem { inner: item })
    }
}

#[pyclass(name = "GSSPopperItem")]
pub struct PyGSSPopperItem {
    inner: GSSPopperItem,
}

#[pymethods]
impl PyGSSPopperItem {
    fn peek_iter(&self) -> Vec<PyGSSPopperItemPeek> {
        self.inner.peek_iter().map(|peek| PyGSSPopperItemPeek { inner: peek }).collect()
    }
}

#[pyclass(name = "GSSPopperItemPeek")]
#[derive(Clone)]
pub struct PyGSSPopperItemPeek {
    inner: GSSPopperItemPeek,
}

#[pymethods]
impl PyGSSPopperItemPeek {
    fn edge_value(&self) -> usize {
        self.inner.edge_value().state_id.0
    }

    fn isolated_parent(&self) -> PyGSSNode {
        PyGSSNode { inner: self.inner.isolated_parent() }
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
    // m.add_class::<PyGLRParser>()?; // Not exposed directly for now
    m.add_class::<PyGrammarConstraint>()?;
    m.add_class::<PyGrammarConstraintState>()?;
    m.add_class::<PyHybridBitset>()?;
    m.add_class::<PyGSSNode>()?;
    m.add_class::<PyGSSPopper>()?;
    m.add_class::<PyGSSPopperItem>()?;
    m.add_class::<PyGSSPopperItemPeek>()?;
    m.add_class::<PyIncrementalParser>()?;
    Ok(())
}
