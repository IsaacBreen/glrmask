//! Forced token computation.
//!
//! When only one token is allowed by the mask, it can be "forced" without
//! sampling. This module provides utilities for detecting and returning
//! forced tokens.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::ds::bitset::BitSet;
use crate::runtime::state::ConstraintState;

/// Check if the mask allows exactly one token. Returns it if so.
#[allow(dead_code)]
pub(crate) fn forced_token(mask: &BitSet) -> Option<u32> {
    unimplemented!()
}

/// Check if the mask is empty (no tokens allowed).
#[allow(dead_code)]
pub(crate) fn is_dead(mask: &BitSet) -> bool {
    unimplemented!()
}

impl<'a> ConstraintState<'a> {
    /// Return the sequence of tokens forced by the current grammar state.
    ///
    /// A token is *forced* when it is the only non-EOS option in the mask.
    /// The method repeatedly computes the mask, collects any single forced
    /// token, simulates a commit, and continues until the state is no longer
    /// deterministic. Returns an empty `Vec` when no tokens are forced.
    ///
    /// The caller is responsible for committing the returned tokens via
    /// [`commit_tokens`].
    pub fn force(&self) -> Vec<u32> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Constraint;
    use crate::Vocab;

    fn make_vocab(entries: &[&str]) -> Vocab {
        let entries: Vec<(u32, Vec<u8>)> = entries
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
            .collect();
        Vocab::new(entries, None)
    }

    #[test]
    fn test_forced_token_detection() {
        let vocab = make_vocab(&["a", "b"]);
        let c = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
        let s = c.start();
        let mask = s.compute_mask();

        // Only "a" should be valid — forced token should be 0.
        assert_eq!(forced_token(&mask), Some(0));
        assert!(!is_dead(&mask));
    }

    #[test]
    fn test_forced_single() {
        let mut mask = BitSet::new(10);
        mask.set(5);
        assert_eq!(forced_token(&mask), Some(5));
    }

    #[test]
    fn test_forced_multi() {
        let mut mask = BitSet::new(10);
        mask.set(5);
        mask.set(7);
        assert_eq!(forced_token(&mask), None);
    }

    #[test]
    fn test_forced_empty() {
        let mask = BitSet::new(10);
        assert_eq!(forced_token(&mask), None);
        assert!(is_dead(&mask));
    }
}
