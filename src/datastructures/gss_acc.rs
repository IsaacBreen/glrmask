use crate::datastructures::leveled_gss::Merge as LGMerge;
use std::collections::{BTreeMap, BTreeSet};

/// Type alias for the disallowed terminals map.
/// Maps tokenizer state ID -> set of disallowed terminal IDs.
pub type TerminalsDisallowed = BTreeMap<usize, BTreeSet<usize>>;

/// Helper function to create a fresh (empty) TerminalsDisallowed.
pub fn terminals_disallowed_fresh() -> TerminalsDisallowed {
    BTreeMap::new()
}

/// Implement Merge for TerminalsDisallowed.
impl LGMerge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, v) in other {
            result.entry(*k).or_default().extend(v.iter().cloned());
        }
        result
    }
}
