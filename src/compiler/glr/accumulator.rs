use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use crate::ds::leveled_gss::Merge;

/// Accumulator stored in the GSS.  Wraps the underlying BTreeMap in an Arc
/// so that Clone is O(1) (reference-count increment) instead of O(n).
/// Mutation is done by cloning the inner map on write.
#[derive(Clone, Debug)]
pub struct TerminalsDisallowed(pub(crate) Arc<BTreeMap<u32, BTreeSet<u32>>>);

impl TerminalsDisallowed {
    pub fn new() -> Self {
        TerminalsDisallowed(Arc::new(BTreeMap::new()))
    }

    /// Return a new TerminalsDisallowed with an additional entry inserted.
    pub fn with_insert(&self, state: u32, terminal: u32) -> Self {
        let mut inner = (*self.0).clone();
        inner.entry(state).or_default().insert(terminal);
        TerminalsDisallowed(Arc::new(inner))
    }
}

/// Deref to BTreeMap for transparent read-only access to all BTreeMap methods.
impl Deref for TerminalsDisallowed {
    type Target = BTreeMap<u32, BTreeSet<u32>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for TerminalsDisallowed {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl Eq for TerminalsDisallowed {}

impl Hash for TerminalsDisallowed {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        if Arc::ptr_eq(&self.0, &other.0) {
            return self.clone();
        }
        let mut merged = (*self.0).clone();
        for (state, terminals) in other.0.iter() {
            merged
                .entry(*state)
                .or_default()
                .extend(terminals.iter().copied());
        }
        TerminalsDisallowed(Arc::new(merged))
    }
}
