//! Graph-Structured Stack (GSS) — simplified list-of-stacks.
//!
//! In GLR parsing, nondeterminism creates multiple parse stacks that may
//! share common prefixes. A full GSS represents this as a DAG, but for
//! simplicity we start with an explicit list of stacks (Vec of Vec).
//!
//! Each "stack" is a sequence of parser state IDs, bottom to top.
//! The top of the stack is `stack.last()`.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

#![allow(dead_code)]

use std::collections::BTreeSet;

/// A collection of GLR parse stacks for a single tokenizer state.
///
/// Each stack is a Vec<u32> of parser state IDs, ordered bottom-to-top.
#[derive(Debug, Clone)]
pub struct GlrStacks {
    stacks: Vec<Vec<u32>>,
}

impl GlrStacks {
    /// Create a new GlrStacks with a single stack containing one state.
    pub fn new(initial_state: u32) -> Self {
        unimplemented!("cargo-check-only stub")
    }

    /// Create from a list of stacks.
    pub fn from_stacks(stacks: Vec<Vec<u32>>) -> Self {
        unimplemented!("cargo-check-only stub")
    }

    /// Create empty (no stacks).
    pub fn empty() -> Self {
        unimplemented!("cargo-check-only stub")
    }

    /// Whether there are no stacks (all paths died).
    pub fn is_empty(&self) -> bool {
        unimplemented!("cargo-check-only stub")
    }

    /// Number of active stacks.
    pub fn len(&self) -> usize {
        unimplemented!("cargo-check-only stub")
    }

    /// Iterate over stacks.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<u32>> {
        std::iter::empty()
    }

    /// Get the top parser state of each stack.
    pub fn tops(&self) -> BTreeSet<u32> {
        unimplemented!("cargo-check-only stub")
    }

    /// Add a stack.
    pub fn push(&mut self, stack: Vec<u32>) {
        unimplemented!("cargo-check-only stub")
    }

    /// Merge with another GlrStacks (union of stacks).
    /// Deduplicates identical stacks.
    pub fn merge(&mut self, other: &GlrStacks) {
        unimplemented!("cargo-check-only stub")
    }

    /// Take ownership of the stacks.
    pub fn into_stacks(self) -> Vec<Vec<u32>> {
        unimplemented!("cargo-check-only stub")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glr_stacks_basic() {
        let gs = GlrStacks::new(0);
        assert_eq!(gs.len(), 1);
        assert!(!gs.is_empty());
        assert_eq!(gs.tops(), BTreeSet::from([0]));
    }

    #[test]
    fn test_glr_stacks_merge_dedup() {
        let mut gs1 = GlrStacks::from_stacks(vec![vec![0, 1], vec![0, 2]]);
        let gs2 = GlrStacks::from_stacks(vec![vec![0, 1], vec![0, 3]]);
        gs1.merge(&gs2);
        assert_eq!(gs1.len(), 3); // [0,1], [0,2], [0,3] — deduped [0,1]
    }
}
