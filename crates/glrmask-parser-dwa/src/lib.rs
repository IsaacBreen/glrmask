#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub use glrmask_vocab::Vocab;

pub mod automata {
    pub use glrmask_finite_automata::automata::unweighted_u32;
    pub use glrmask_weighted_automata::automata::weighted_u32;
    pub use weighted_u32 as weighted;
}

pub mod ds {
    pub use glrmask_lexer::ds::bitset;
    pub use glrmask_weight as weight;
}

pub mod grammar {
    pub use glrmask_grammar::grammar::*;
}

pub mod runtime {
    pub use glrmask_artifact::CommitTemplateDfas;
}

pub mod compiler {
    pub use glrmask_glr::glr;

    pub mod stages {
        pub use glrmask_artifact::equiv_types;
        pub use crate::parser_dwa;
        pub use crate::resolve_negatives;
        pub use crate::templates;
    }
}

pub mod merge;
pub mod parser_dwa;
pub mod resolve_negatives;
pub mod templates;
