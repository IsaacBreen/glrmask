//! Runtime action helpers.
//!
//! These modules own the sequence actions exposed through `ConstraintState`:
//! mask computation, token/byte commit, and forced-token discovery.

pub(crate) mod commit;
pub(crate) mod force;
pub(crate) mod mask;
