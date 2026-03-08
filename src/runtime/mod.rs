#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

// SEP1_MAP: `runtime` is the glrmask split-out analogue of sep1's mostly
// monolithic runtime surface in `grammars2024/src/constraint.rs` plus
// `grammars2024/src/constraint_fns.rs`.
mod actions;
mod constraint;
mod debug;
mod serde;
mod state;


pub use constraint::Constraint;
pub use state::ConstraintState;
