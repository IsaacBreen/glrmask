//! Renderers for grammar IR.
//!
//! Renderers are observational. They may inspect grammar syntax, but they must
//! not allocate compiler ids, emit productions, or mutate the grammar.

pub mod glrm;
pub mod lark;
