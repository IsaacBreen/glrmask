#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) use glrmask_vocab::Vocab;

pub(crate) mod ds {
    pub(crate) use glrmask_vocab::__private::vocab_prefix_tree;
    pub(crate) mod bitset;
    pub(crate) mod char_transitions;
    pub(crate) mod compressed_state_set;
    pub(crate) mod u8set;
}

pub(crate) mod grammar {
    pub(crate) mod flat {
        pub(crate) type TerminalID = u32;
    }
}

pub(crate) mod automata {
    pub(crate) use glrmask_finite_automata::unweighted_u32;
    pub(crate) mod lexer;
    pub(crate) use lexer::ast as regex;
}

mod possible_matches;

pub use automata::lexer::ast::Expr;
pub use automata::lexer::dfa::DFA;
pub use automata::lexer::regex::parse_regex;
pub use ds::u8set::U8Set;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub mod ds {
        pub mod bitset {
            pub use crate::ds::bitset::*;
        }
        pub mod char_transitions {
            pub use crate::ds::char_transitions::*;
        }
        pub mod compressed_state_set {
            pub use crate::ds::compressed_state_set::*;
        }
        pub mod u8set {
            pub use crate::ds::u8set::*;
        }
        pub mod vocab_prefix_tree {
            pub use glrmask_vocab::__private::vocab_prefix_tree::*;
        }
    }

    pub mod automata {
        pub use glrmask_finite_automata::unweighted_u32;

        pub mod lexer {
            pub use crate::automata::lexer::{DFA, Lexer};
            pub mod ast {
                pub use crate::automata::lexer::ast::*;
            }
            pub mod compile {
                pub use crate::automata::lexer::compile::*;
            }
            pub mod tokenizer {
                pub use crate::automata::lexer::tokenizer::*;
            }
            pub mod regex {
                pub use crate::automata::lexer::regex::*;
            }
        }

        pub mod regex {
            pub use crate::automata::lexer::ast::*;
        }
    }

    pub mod possible_matches {
        pub use crate::possible_matches::*;
    }
}
