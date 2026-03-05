//! Graph-Structured Stack (GSS) — simplified list-of-stacks.
//!
//! In GLR parsing, nondeterminism creates multiple parse stacks that may
//! share common prefixes. A full GSS represents this as a DAG, but for
//! simplicity we start with an explicit list of stacks (Vec of Vec).
//!
//! Each "stack" is a sequence of parser state IDs, bottom to top.
//! The top of the stack is `stack.last()`.

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
        Self {
            stacks: vec![vec![initial_state]],
        }
    }

    /// Create from a list of stacks.
    pub fn from_stacks(stacks: Vec<Vec<u32>>) -> Self {
        Self { stacks }
    }

    /// Create empty (no stacks).
    pub fn empty() -> Self {
        Self { stacks: Vec::new() }
    }

    /// Whether there are no stacks (all paths died).
    pub fn is_empty(&self) -> bool {
        self.stacks.is_empty()
    }

    /// Number of active stacks.
    pub fn len(&self) -> usize {
        self.stacks.len()
    }

    /// Iterate over stacks.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<u32>> {
        self.stacks.iter()
    }

    /// Get the top parser state of each stack.
    pub fn tops(&self) -> BTreeSet<u32> {
        self.stacks
            .iter()
            .filter_map(|s| s.last().copied())
            .collect()
    }

    /// Add a stack.
    pub fn push(&mut self, stack: Vec<u32>) {
        self.stacks.push(stack);
    }

    /// Merge with another GlrStacks (union of stacks).
    /// Deduplicates identical stacks.
    pub fn merge(&mut self, other: &GlrStacks) {
        let existing: BTreeSet<Vec<u32>> = self.stacks.drain(..).collect();
        let mut all: BTreeSet<Vec<u32>> = existing;
        for s in &other.stacks {
            all.insert(s.clone());
        }
        self.stacks = all.into_iter().collect();
    }

    /// Take ownership of the stacks.
    pub fn into_stacks(self) -> Vec<Vec<u32>> {
        self.stacks
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
