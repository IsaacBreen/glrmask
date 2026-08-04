#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) use glrmask_vocab::Vocab;

pub(crate) mod automata {
    pub(crate) use glrmask_finite_automata::unweighted_u32;
    pub(crate) use glrmask_weighted_automata::weighted_u32;
    pub(crate) use weighted_u32 as weighted;
}

pub(crate) mod ds {
    pub(crate) use glrmask_lexer::__private::ds::bitset;
    pub(crate) use glrmask_weight::__private as weight;
}

pub(crate) mod grammar {
    pub(crate) use glrmask_grammar::__private::grammar::*;
}

pub(crate) mod runtime {
    pub(crate) use glrmask_artifact::__private::CommitTemplateDfas;
}

pub(crate) mod compiler {
    pub(crate) use glrmask_glr::__private::glr;

    pub(crate) mod stages {
        pub(crate) use crate::resolve_negatives;
        pub(crate) use crate::templates;
        pub(crate) use glrmask_artifact::__private::equiv_types;
    }
}

#[cfg(feature = "internal-api")]
pub(crate) mod merge;
pub(crate) mod parser_dwa;
pub(crate) mod resolve_negatives;
pub(crate) mod templates;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod merge {
        pub use crate::merge::*;
    }
    pub mod parser_dwa {
        pub use crate::parser_dwa::*;
    }
    pub mod resolve_negatives {
        pub use crate::resolve_negatives::*;
    }
    pub mod templates {
        pub use crate::templates::*;
        pub mod characterize {
            pub use crate::templates::characterize::*;
        }
        pub mod compile_bundle {
            pub use crate::templates::compile_bundle::*;
        }
        pub mod compile_dfa {
            pub use crate::templates::compile_dfa::*;
        }
    }
}
