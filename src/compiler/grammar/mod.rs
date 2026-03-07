

// SEP1_MAP: glrmask keeps a compiler-local grammar layer here; sep1 spreads the comparable grammar-definition surface across interface/interface.rs and glr/grammar.rs.

pub mod model;
pub mod normalize;

pub use model::*;
pub use normalize::*;