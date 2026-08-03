#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod automata {
    pub(crate) use glrmask_weighted_automata::weighted_u32;
}

pub(crate) mod ds {
    pub(crate) use glrmask_weight::__private as weight;
}

mod commit_templates;
pub(crate) mod equiv_types;
pub(crate) mod mapped_artifact;

pub use commit_templates::CommitTemplateDfas;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod equiv_types {
        pub use crate::equiv_types::*;
    }
    pub mod mapped_artifact {
        pub use crate::mapped_artifact::*;
    }
    pub use crate::commit_templates::CommitTemplateDfas;
}
