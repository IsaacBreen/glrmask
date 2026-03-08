// SEP1_MAP: glrmask's compiler debug module has no single sep1 counterpart; related display and inspection code is scattered across glr/parser.rs, glr/stats.rs, and precompute diagnostics.

pub mod artifacts;
mod display;

pub use artifacts::*;
