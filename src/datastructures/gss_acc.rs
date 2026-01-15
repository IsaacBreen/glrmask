use crate::datastructures::leveled_gss::Merge as LGMerge;
use std::collections::{BTreeMap, BTreeSet};

/// Type alias for terminals disallowed at each tokenizer state.
/// Maps tokenizer_state_id -> set of terminal IDs that are disallowed.
pub type TerminalsDisallowed = BTreeMap<usize, BTreeSet<usize>>;

/// Merge implementation for TerminalsDisallowed (combines disallowed terminal sets).
impl LGMerge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, v) in other {
            result.entry(*k).or_default().extend(v);
        }
        result
    }
}

/// Create a fresh (empty) TerminalsDisallowed - no terminals disallowed yet.
pub fn terminals_disallowed_fresh() -> TerminalsDisallowed {
    BTreeMap::new()
}
