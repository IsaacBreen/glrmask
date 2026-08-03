#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub mod automata {
    pub use glrmask_weighted_automata::automata::weighted_u32;
    pub use weighted_u32 as weighted;
}

pub mod compiler {
    pub use glrmask_glr::glr;

    pub mod stages {
        pub use glrmask_artifact::{equiv_types, mapped_artifact};
    }
}

pub mod ds {
    pub use glrmask_weight as weight;
}

pub mod merge;
pub mod types;

pub use merge::*;
pub use types::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile};
