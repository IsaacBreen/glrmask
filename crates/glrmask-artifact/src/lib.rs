#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub mod automata {
    pub use glrmask_weighted_automata::automata::weighted_u32;
}

pub mod ds {
    pub use glrmask_weight as weight;
}

pub mod commit_templates;
pub mod equiv_types;
pub mod mapped_artifact;

pub mod compiler {
    pub mod stages {
        pub use crate::equiv_types;
        pub use crate::mapped_artifact;
    }
}

pub use commit_templates::CommitTemplateDfas;
