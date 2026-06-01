//! Public parser/grammar introspection types.
//!
//! These are not part of the mask/commit hot path.  They are exposed because the
//! compiled constraint can report parser-table ambiguity and terminal display
//! names, which are useful when validating a grammar against the paper model.

#[doc(inline)]
pub use crate::parser::glr::table::{TableAmbiguity, TableAmbiguityKind};
