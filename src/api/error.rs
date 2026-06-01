//! Public error and result aliases.
//!
//! These are deliberately coarse today: grammar import, compilation, and
//! serialization.  A later error-policy pass should preserve these stable
//! top-level categories while adding more structured context internally.

#[doc(inline)]
pub use crate::error::{Error, GlrMaskError, Result};
