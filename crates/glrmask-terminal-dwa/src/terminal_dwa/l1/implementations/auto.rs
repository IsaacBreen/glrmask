//! Compatibility alias for projected L1's internal kernel selector.
//!
//! The historical projected-vs-quotient auto selector is retired: production
//! never routes to the quotient implementation.

use super::{BuildInput, LocalIdMapTerminalDwa};

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    super::projected::build(input)
}
