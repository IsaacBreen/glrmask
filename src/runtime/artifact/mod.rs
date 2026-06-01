//! Runtime artifact namespace.
//!
//! This module separates the immutable compiled data from the derived runtime
//! caches used to make Mask and Commit fast.
//!
//! Mathematical reading order:
//!
//! 1. `compiled` — the serialized semantic object, currently public as
//!    `Constraint`;
//! 2. `token_space` — maps between original token ids, final internal token ids,
//!    original lexer states, and final internal lexer-state ids;
//! 3. `templates` — parser stack-effect recognizers used by Commit;
//! 4. `cache_types` and `caches` — derived materialization and transition
//!    caches rebuilt after compile/load;
//! 5. `serialization` — the versioned external representation.

mod accessors;
mod cache_types;
mod caches;
mod compiled;
mod dense;
mod finalize;
mod serialization;
mod templates;
mod token_space;

pub use compiled::Constraint;
pub(crate) use compiled::CompiledArtifactParts;
pub(crate) use templates::{CommitTemplateDfas, TemplateDfasByTerminal};
