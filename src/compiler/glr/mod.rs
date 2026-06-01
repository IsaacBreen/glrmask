//! Compatibility shim for the historical `compiler::glr` path.
//!
//! The GLR machinery is now owned by `crate::parser::glr`, because it is not
//! only a compiler implementation detail: the runtime uses the same parser
//! stack-advance semantics for Mask and Commit.  New code should import from
//! `crate::parser::glr::*` directly.

#[doc(hidden)]
pub use crate::parser::glr::*;

#[doc(hidden)]
pub mod accumulator {
    pub use crate::parser::glr::accumulator::*;
}

#[doc(hidden)]
pub mod analysis {
    pub use crate::parser::glr::analysis::*;
}

#[doc(hidden)]
pub mod labels {
    pub use crate::parser::glr::labels::*;
}

#[doc(hidden)]
pub mod parser {
    pub use crate::parser::glr::advance::*;
}

#[doc(hidden)]
pub mod table {
    pub use crate::parser::glr::table::*;
}
