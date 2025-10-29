use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use crate::constraint::StageVocab;
use crate::glr::parser::GLRParser;

/// Shared, immutable and mutable context passed to every optimization pass.
/// Keeps global bounds and simple telemetry.
#[derive(Clone)]
pub struct OptimizationContext<'a> {
    pub max_llm_token_id: usize,
    pub max_state_id: usize,
    pub iteration_budget: usize,
    pub debug_level: u8,
    pub assert_no_pop0_except_roots: bool,
    pub metrics_before: BTreeMap<String, String>,
    pub metrics_after: BTreeMap<String, String>,
    // New fields for advanced passes
    pub stage_vocab: Option<Rc<RefCell<&'a mut StageVocab>>>,
    pub parser: Option<Rc<RefCell<&'a GLRParser>>>,
}

impl<'a> OptimizationContext<'a> {
    pub fn new(max_llm_token_id: usize, max_state_id: usize) -> Self {
        Self {
            max_llm_token_id,
            max_state_id,
            iteration_budget: 1_000_000,
            debug_level: 1,
            assert_no_pop0_except_roots: true,
            metrics_before: BTreeMap::new(),
            metrics_after: BTreeMap::new(),
            stage_vocab: None,
            parser: None,
        }
    }
}

impl<'a> fmt::Debug for OptimizationContext<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptimizationContext")
            .field("max_llm_token_id", &self.max_llm_token_id)
            .field("max_state_id", &self.max_state_id)
            .field("iteration_budget", &self.iteration_budget)
            .field("debug_level", &self.debug_level)
            .field("assert_no_pop0_except_roots", &self.assert_no_pop0_except_roots)
            .field("metrics_before", &self.metrics_before)
            .field("metrics_after", &self.metrics_after)
            .field("stage_vocab_present", &self.stage_vocab.is_some())
            .field("parser_present", &self.parser.is_some())
            .finish()
    }
}

