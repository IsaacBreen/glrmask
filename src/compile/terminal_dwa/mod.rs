//! Terminal DWA construction.
//!
//! The Terminal DWA is the weighted automaton over completed grammar-terminal
//! sequences.  Evaluating it on a terminal sequence returns the lexer-state /
//! token pairs whose token bytes can emit exactly that sequence.
//!
//! The publication-facing structure of this module is deliberately mathematical:
//!
//! 1. [`options`] reads historical environment knobs and names them as build
//!    policy, not as part of the denotation.
//! 2. [`vocab_partition`] chooses a vocabulary partitioning strategy.  This is
//!    a performance quotient over caller tokens; it is not the terminal DWA
//!    itself.
//! 3. [`direct_partition`] builds the direct/single-step component.
//! 4. [`pair_partition`] builds the multi-step component.
//! 5. [`merge`] reconciles local id maps and local automata into one global
//!    Terminal DWA artifact.
//! 6. [`global_state_map`] computes the tokenizer-state quotient shared by the
//!    Terminal-DWA and scan-relation phases.
//! 7. [`builder`] is the only top-level orchestration layer.
//!
//! This split keeps the central denotation simple: the Terminal DWA accepts a
//! terminal sequence and returns a mask over `(lexer state, token)` pairs.  All
//! partitioning, quotienting, caching, and profiling exists to build that object
//! faster or smaller without changing that denotation.

pub(crate) mod builder;
pub(crate) mod classify;
pub(crate) mod direct_partition;
pub(crate) mod global_state_map;
pub(crate) mod grammar_helpers;
pub(crate) mod merge;
pub(crate) mod options;
pub(crate) mod pair_partition;
pub(crate) mod partition;
pub(crate) mod types;
pub(crate) mod vocab_partition;

pub(crate) use builder::{
    build_terminal_dwa,
    build_terminal_dwa_with_precomputed_global_max_length,
};
pub(crate) use global_state_map::build_global_max_length_state_map;
pub(crate) use vocab_partition::prepare_vocab_for_terminal_dwa;
