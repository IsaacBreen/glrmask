#![deny(warnings)]
#![allow(dead_code)]

mod implementation;

pub use implementation::{SharedTokenSet, Weight};

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub use crate::implementation::*;
}
