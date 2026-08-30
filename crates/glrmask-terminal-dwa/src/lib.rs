#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

use thiserror::Error as ThisError;

pub(crate) use glrmask_vocab::__private as vocab;
pub(crate) use glrmask_vocab::Vocab;

#[derive(ThisError, Debug)]
pub(crate) enum Error {
    #[error("Internal compiler invariant violated: {0}")]
    InternalInvariant(String),
}

pub(crate) mod error {
    pub(crate) use super::Error;
    pub(crate) use glrmask_invariant::__private::fail_internal_invariant;

    pub(crate) fn catch_internal_invariant<T>(f: impl FnOnce() -> T) -> Result<T, Error> {
        glrmask_invariant::__private::catch_internal_invariant_message(f).map_err(Error::InternalInvariant)
    }
}

pub(crate) mod automata {
    pub(crate) use glrmask_finite_automata::unweighted_u32;
    pub(crate) use glrmask_lexer::__private::automata::{lexer, regex};
    pub(crate) use glrmask_weighted_automata::weighted_u32;
    pub(crate) use weighted_u32 as weighted;
}

pub(crate) mod ds {
    pub(crate) use glrmask_lexer::__private::ds::{bitset, u8set};
    pub(crate) use glrmask_vocab::__private::vocab_prefix_tree;
    pub(crate) use glrmask_weight::__private as weight;
}

pub(crate) mod grammar {
    pub(crate) use glrmask_grammar::__private::grammar::*;
}

pub(crate) mod compiler {
    pub(crate) use glrmask_glr::__private::glr;
    pub(crate) use glrmask_lexer::__private::possible_matches;

    pub(crate) mod stages {
        pub(crate) use crate::terminal_dwa as id_map_and_terminal_dwa;
        pub(crate) use glrmask_artifact::__private::{equiv_types, mapped_artifact};
    }
}

pub mod terminal_dwa;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod terminal_dwa {
        pub use crate::terminal_dwa::*;
    }
}
