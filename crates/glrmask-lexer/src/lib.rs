#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub use glrmask_vocab::Vocab;

pub mod ds {
    pub mod bitset;
    pub mod char_transitions;
    pub mod compressed_state_set;
    pub mod u8set;
}

pub mod grammar {
    pub mod flat {
        pub type TerminalID = u32;
    }
}

pub mod automata {
    pub use glrmask_finite_automata::automata::unweighted_u32;
    pub mod lexer;
    pub use lexer::ast as regex;
}

pub use automata::lexer;
