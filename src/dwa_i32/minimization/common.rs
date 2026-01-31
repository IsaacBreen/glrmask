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
    ExactMinimize,
    RustfstMinimize,
    ConsolidateRanges,
    TrimWeights,
}

impl DwaPass {
    pub fn is_enabled(&self) -> bool {
        match self {
            DwaPass::PruneUnreachable => std::env::var("DWA_DISABLE_PRUNE_UNREACHABLE").map(|v| v != "1").unwrap_or(true),
            DwaPass::PruneDeadEnds => std::env::var("DWA_DISABLE_PRUNE_DEAD_ENDS").map(|v| v != "1").unwrap_or(true),
            DwaPass::PushWeights => std::env::var("DWA_DISABLE_PUSH_WEIGHTS").map(|v| v != "1").unwrap_or(true),
            DwaPass::PushWeightsToInitial => std::env::var("DWA_DISABLE_PUSH_WEIGHTS_TO_INITIAL").map(|v| v != "1").unwrap_or(true),
            DwaPass::ResidualPush => std::env::var("DWA_DISABLE_RESIDUAL_PUSH").map(|v| v != "1").unwrap_or(true),
            DwaPass::ExactMinimize => std::env::var("DWA_DISABLE_MINIMIZE").map(|v| v != "1").unwrap_or(true),
            DwaPass::RustfstMinimize => std::env::var("DWA_DISABLE_RUSTFST_MINIMIZE").map(|v| v != "1").unwrap_or(true),
            // ConsolidateRanges is slow in weight-heavy mode due to large weight domain
            // Disabled by default in weight-heavy mode (num_tsids > 1)
            DwaPass::ConsolidateRanges => {
                if std::env::var("DWA_DISABLE_CONSOLIDATE_RANGES").map(|v| v == "1").unwrap_or(false) {
                    return false;
                }
                // Check if weight-heavy mode is enabled (num_tsids > 1)
                let num_tsids = crate::datastructures::get_num_tsids();
                if num_tsids > 1 {
                    // Disabled by default in weight-heavy mode, can be explicitly enabled
                    std::env::var("DWA_ENABLE_CONSOLIDATE_RANGES_WEIGHT_HEAVY").map(|v| v == "1").unwrap_or(false)
                } else {
                    true // Enabled by default in symbol-heavy mode
                }
            },
            // TrimWeights clips ranges to actual max values, removing unnecessary usize::MAX extensions
            DwaPass::TrimWeights => std::env::var("DWA_DISABLE_TRIM_WEIGHTS").map(|v| v != "1").unwrap_or(true),
        }
    }
}
