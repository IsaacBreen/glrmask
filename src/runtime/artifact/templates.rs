//! Commit-time parser stack-effect recognizers.
//!
//! These DFAs are derived from the parser table during compilation.  They are
//! runtime accelerators for Commit, but mathematically they represent the same
//! terminal stack-effect relation consumed by Parser-DWA construction.

use std::sync::Arc;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;

/// Template DFA triple for one grammar terminal.
#[derive(Debug, Clone, Default)]
pub(crate) struct CommitTemplateDfas {
    pub(crate) pop: UnweightedDfa,
    pub(crate) read: UnweightedDfa,
    pub(crate) push: UnweightedDfa,
    pub(crate) pop_to_read: Vec<Option<u32>>,
    pub(crate) pop_to_push: Vec<Option<u32>>,
    pub(crate) read_to_push: Vec<Option<u32>>,
}

/// Template DFA bundle indexed by grammar terminal id.
pub(crate) type TemplateDfasByTerminal = Vec<Option<Arc<CommitTemplateDfas>>>;
