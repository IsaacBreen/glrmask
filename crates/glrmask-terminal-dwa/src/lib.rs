#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

use thiserror::Error as ThisError;

pub use glrmask_vocab::Vocab;
pub use glrmask_vocab as vocab;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Internal compiler invariant violated: {0}")]
    InternalInvariant(String),
}

pub mod error {
    pub use super::Error;
    pub use glrmask_invariant::fail_internal_invariant;

    pub fn catch_internal_invariant<T>(f: impl FnOnce() -> T) -> Result<T, Error> {
        glrmask_invariant::catch_internal_invariant_message(f).map_err(Error::InternalInvariant)
    }
}

pub mod automata {
    pub use glrmask_finite_automata::automata::unweighted_u32;
    pub use glrmask_lexer::automata::{lexer, regex};
    pub use glrmask_weighted_automata::automata::weighted_u32;
    pub use weighted_u32 as weighted;
}

pub mod ds {
    pub use glrmask_lexer::ds::{bitset, u8set};
    pub use glrmask_vocab::vocab_prefix_tree;
    pub use glrmask_weight as weight;
}

pub mod grammar {
    pub use glrmask_grammar::grammar::*;
}

pub mod compiler {
    pub use glrmask_glr::glr;
    pub use glrmask_lexer::possible_matches;

    pub mod stages {
        pub use glrmask_artifact::{equiv_types, mapped_artifact};
        pub use crate::terminal_dwa as id_map_and_terminal_dwa;
    }
}

pub mod terminal_dwa;
