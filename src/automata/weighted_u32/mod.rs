#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

// SEP1_MAP: This module is the glrmask counterpart to sep1's weighted parser automata in `dwa_i32/`.

pub mod determinize;
pub mod dwa;
pub mod minimize;
pub mod nwa;

#[cfg(test)]
mod test_weighted_automata;

#[cfg(test)]
mod test_determinization;

#[cfg(test)]
mod test_weight_loosening;
