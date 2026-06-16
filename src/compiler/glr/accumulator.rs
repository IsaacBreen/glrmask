use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use crate::ds::leveled_gss::Merge;

/// Accumulator stored in the GSS.  Wraps the underlying BTreeMap in an Arc
/// so that Clone is O(1) (reference-count increment) instead of O(n).
/// Mutation is done by cloning the inner map on write.
#[derive(Clone, Debug)]
pub struct TerminalsDisallowed(pub(crate) Arc<BTreeMap<usize, BTreeSet<u32>>>);

impl TerminalsDisallowed {
    pub fn new() -> Self {
        TerminalsDisallowed(Arc::new(BTreeMap::new()))
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        if self.0.len() > other.0.len() {
            return false;
        }
        for (state, terminals) in self.0.iter() {
            let Some(other_terminals) = other.0.get(state) else {
                return false;
            };
            if !terminals.is_subset(other_terminals) {
                return false;
            }
        }
        true
    }

    /// Return a new TerminalsDisallowed with an additional entry inserted.
    pub fn with_insert(&self, state: usize, terminal: u32) -> Self {
        let mut inner = (*self.0).clone();
        inner.entry(state).or_default().insert(terminal);
        TerminalsDisallowed(Arc::new(inner))
    }
}

/// Deref to BTreeMap for transparent read-only access to all BTreeMap methods.
impl Deref for TerminalsDisallowed {
    type Target = BTreeMap<usize, BTreeSet<u32>>;
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
        if self.is_subset_of(other) {
            return other.clone();
        }
        if other.is_subset_of(self) {
            return self.clone();
        }

        let (base, extra) = if self.0.len() >= other.0.len() {
            (&self.0, &other.0)
        } else {
            (&other.0, &self.0)
        };
        let mut merged = (**base).clone();
        for (state, terminals) in extra.iter() {
            merged
                .entry(*state)
                .or_default()
                .extend(terminals.iter().copied());
        }
        TerminalsDisallowed(Arc::new(merged))
    }

    fn subsumes(&self, other: &Self) -> bool {
        other.is_subset_of(self)
    }
}
