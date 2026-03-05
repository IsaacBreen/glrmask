//! DWA optimization passes.
//!
//! Post-processing passes on the compiled DWA:
//! - Dead state elimination
//! - Weight normalization
//! - State renumbering

use crate::automata::weighted::dwa::Dwa;

/// Apply all optimization passes to a DWA.
pub fn optimize(dwa: Dwa) -> Dwa {
    // TODO: Implement optimization passes
    dwa
}
