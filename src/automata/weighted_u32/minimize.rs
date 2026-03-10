//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to minimize.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use super::dwa::DWA;
use crate::ds::weight::Weight;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StateSignature {
    final_weight: Option<Weight>,
    transitions: Vec<(i32, usize, Weight)>,
}

pub fn minimize(dwa: &DWA) -> DWA {
    if dwa.states.is_empty() {
        return dwa.clone();
    }
    if !dwa.is_acyclic() {
        return dwa.clone();
    }

    // Use graph-coloring minimizer
    super::minimize_acyclic::minimize_acyclic(dwa)
}
