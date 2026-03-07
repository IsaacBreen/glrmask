//! Placeholder weighted minimization surface.
//!
//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to minimize.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use super::dwa::DWA;

/// Minimize an acyclic [`DWA`].
///
/// Cyclic input should panic.
pub fn minimize(_dwa: &DWA) -> DWA {
    todo!("weighted minimization is intentionally deferred and must remain acyclic-only")
}
