//! DWA optimization passes.
//!
//! Post-processing passes on the compiled DWA:
//! - Dead state elimination
//! - Weight normalization
//! - State renumbering
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

use crate::automata::weighted::dwa::Dwa;

#[allow(dead_code)]
    /// Apply all optimization passes to a DWA.
pub fn optimize(dwa: Dwa) -> Dwa {
    unimplemented!("cargo-check-only stub")
}
