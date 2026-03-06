//! Forced token computation.
//!
//! When only one token is allowed by the mask, it can be "forced" without
//! sampling. This module provides utilities for detecting and returning
//! forced tokens.

use crate::ds::bitset::BitSet;

/// Check if the mask allows exactly one token. Returns it if so.
#[allow(dead_code)]
pub(crate) fn forced_token(mask: &BitSet) -> Option<u32> {
    if mask.count_ones() == 1 {
        // Find the single set bit.
        for i in 0..mask.len() {
            if mask.get(i) {
                return Some(i as u32);
            }
        }
    }
    None
}

/// Check if the mask is empty (no tokens allowed).
#[allow(dead_code)]
pub(crate) fn is_dead(mask: &BitSet) -> bool {
    mask.count_ones() == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::state::Constraint;
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
