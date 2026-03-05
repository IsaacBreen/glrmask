//! Forced token computation.
//!
//! When only one token is allowed by the mask, it can be "forced" without
//! sampling. This module provides utilities for detecting and returning
//! forced tokens.

use crate::ds::bitset::BitSet;

/// Check if the mask allows exactly one token. Returns it if so.
pub fn forced_token(mask: &BitSet) -> Option<u32> {
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
pub fn is_dead(mask: &BitSet) -> bool {
    mask.count_ones() == 0
}

#[cfg(test)]
mod tests {
    use super::*;

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
