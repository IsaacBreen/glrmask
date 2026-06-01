//! Parser-domain machinery shared by compile-time construction and runtime queries.
//!
//! The paper treats parser stack behavior abstractly: a grammar frontend supplies
//! terminal stack-effect recognizers; compilation builds the Parser DWA from
//! those recognizers; runtime queries walk concrete parser stack prefixes through
//! that compiled object.  This module owns the parser-side objects that are not
//! themselves one of the paper's named compiled automata.

pub(crate) mod glr;
pub(crate) mod gss;
