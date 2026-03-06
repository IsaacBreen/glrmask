//! GSS accumulator types for tracking disallowed terminals per tokenizer state.
//!
//! During GLR parsing, different parse paths may allow/disallow different
//! terminal matches at different tokenizer states. The `TerminalsDisallowed`
//! accumulator tracks this: `tokenizer_state_id → set of disallowed terminal_ids`.
//!
//! When two GSS branches merge, their disallowed sets are unioned — this is
//! a safe overapproximation that may block some tokens but never incorrectly
//! allows invalid ones.

use std::collections::{BTreeMap, BTreeSet};

use super::leveled_gss::Merge;

/// Maps tokenizer state ID → set of disallowed terminal IDs.
///
/// Used as the GSS accumulator to track which (tsid, terminal) pairs
/// should be excluded during mask computation.
pub type TerminalsDisallowed = BTreeMap<u32, BTreeSet<u32>>;

/// Create a fresh (empty) TerminalsDisallowed — no terminals are disallowed.
pub fn terminals_disallowed_fresh() -> TerminalsDisallowed {
    BTreeMap::new()
}

/// Implement Merge for TerminalsDisallowed.
///
/// When two parse branches merge, the disallowed sets are unioned:
/// if *either* branch disallows a terminal at a tokenizer state,
/// the merged result also disallows it.
impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, v) in other {
            result.entry(*k).or_default().extend(v.iter().cloned());
        }
        result
    }
}
