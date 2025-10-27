use crate::json_serialization::JSONNode;
use std::collections::BTreeMap;
use std::fmt;

/// Shared, immutable and mutable context passed to every optimization pass.
/// Keeps global bounds and simple telemetry.
#[derive(Clone)]
pub struct OptimizationContext {
    pub max_llm_token_id: usize,
    pub max_state_id: usize,
    pub iteration_budget: usize,
    pub debug_level: u8,
    pub metrics_before: BTreeMap<String, JSONNode>,
    pub metrics_after: BTreeMap<String, JSONNode>,
}

impl OptimizationContext {
    pub fn new(max_llm_token_id: usize, max_state_id: usize) -> Self {
        Self {
            max_llm_token_id,
            max_state_id,
            iteration_budget: 1_000_000,
            debug_level: 1,
            metrics_before: BTreeMap::new(),
            metrics_after: BTreeMap::new(),
        }
    }
}

impl fmt::Debug for OptimizationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptimizationContext")
            .field("max_llm_token_id", &self.max_llm_token_id)
            .field("max_state_id", &self.max_state_id)
            .field("iteration_budget", &self.iteration_budget)
            .field("debug_level", &self.debug_level)
            .field("metrics_before", &self.metrics_before)
            .field("metrics_after", &self.metrics_after)
            .finish()
    }
}
