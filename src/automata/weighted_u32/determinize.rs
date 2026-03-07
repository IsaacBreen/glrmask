//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to determinize.
// SEP1_MAP: The closest sep1 analogue is the weighted determinization family under `dwa_i32/determinization*.rs`, narrowed here to the acyclic-only policy.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::dwa::DWA;
use super::nwa::NWA;
use crate::GlrMaskError;

pub fn determinize(_nwa: &NWA) -> Result<DWA, GlrMaskError> {
    todo!("weighted determinization is intentionally deferred and must remain acyclic-only")
}
