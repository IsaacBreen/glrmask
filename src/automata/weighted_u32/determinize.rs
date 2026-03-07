//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to determinize.
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
