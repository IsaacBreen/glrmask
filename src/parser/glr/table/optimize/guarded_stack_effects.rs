//! Guarded stack-effect optimization fragments.
//!
//! The implementation is textually included from the files in
//! `optimize/guarded/` by `table/optimize.rs`.  This file is kept as a
//! reading map for source-tree navigation.
//!
//! - `frame_model.rs`: symbolic stack-effect frames and action keys.
//! - `reduce_frame.rs`: reduction composition over symbolic frames.
//! - `action_exploration.rs`: recursive action exploration into stack effects.
//! - `action_materialize.rs`: conversion from symbolic effects back to table actions.
//! - `stack_shift_canonicalization.rs`: canonicalization of concrete stack shifts.
