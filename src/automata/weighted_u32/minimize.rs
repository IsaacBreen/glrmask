//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to minimize.
// SEP1_MAP: The nearest sep1 analogue is the weighted minimization pipeline under `dwa_i32/minimization/**`, again narrowed here to acyclic-only behavior.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::dwa::DWA;

pub fn minimize(_dwa: &DWA) -> DWA {
    todo!("weighted minimization is intentionally deferred and must remain acyclic-only")
}
