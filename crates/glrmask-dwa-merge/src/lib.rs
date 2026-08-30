#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod automata {
    pub(crate) use glrmask_weighted_automata::weighted_u32;
    pub(crate) use weighted_u32 as weighted;
}

pub(crate) mod compiler {
    pub(crate) use glrmask_glr::__private::glr;

    pub(crate) mod stages {
        pub(crate) use glrmask_artifact::__private::{equiv_types, mapped_artifact};
    }
}

pub(crate) mod ds {
    pub(crate) use glrmask_weight::__private as weight;
}

pub mod merge;
pub use merge::merge_vocab_token_maps;
pub(crate) mod types;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod merge {
        pub use crate::merge::*;
    }
    pub use crate::types::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile};
}
