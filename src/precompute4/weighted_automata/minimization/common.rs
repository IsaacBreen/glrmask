//! Common types shared between DWA and NWA minimization.

pub const MAX_OPTIMIZE_ITERATIONS: usize = 1000;

/// Partition for state minimization.
#[derive(Clone, Debug)]
pub struct Partition {
    pub class_of: Vec<usize>,
    pub num_classes: usize,
}

impl Partition {
    pub fn new(num_states: usize) -> Self {
        Partition {
            class_of: vec![0; num_states],
            num_classes: 1,
        }
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushWeights,
    PushWeightsToInitial,
    ResidualPush,
    Minimize,
    ConsolidateRanges,
}

impl DwaPass {
    pub fn is_enabled(&self) -> bool {
        match self {
            DwaPass::PruneUnreachable => std::env::var("DWA_DISABLE_PRUNE_UNREACHABLE").map(|v| v != "1").unwrap_or(true),
            DwaPass::PruneDeadEnds => std::env::var("DWA_DISABLE_PRUNE_DEAD_ENDS").map(|v| v != "1").unwrap_or(true),
            DwaPass::PushWeights => std::env::var("DWA_DISABLE_PUSH_WEIGHTS").map(|v| v != "1").unwrap_or(true),
            DwaPass::PushWeightsToInitial => std::env::var("DWA_DISABLE_PUSH_WEIGHTS_TO_INITIAL").map(|v| v != "1").unwrap_or(true),
            DwaPass::ResidualPush => std::env::var("DWA_DISABLE_RESIDUAL_PUSH").map(|v| v != "1").unwrap_or(true),
            DwaPass::Minimize => std::env::var("DWA_DISABLE_MINIMIZE").map(|v| v != "1").unwrap_or(true),
            DwaPass::ConsolidateRanges => std::env::var("DWA_ENABLE_CONSOLIDATE_RANGES").map(|v| v == "1").unwrap_or(false),
        }
    }
}
