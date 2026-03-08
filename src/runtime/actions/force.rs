//! Forced-token computation.
//!
//! Mirrors sep1's `force()` algorithm:
//!
//! 1. **Compute forced byte prefix** — at each step, check whether all tokens
//!    allowed by the mask share the same first byte.  If yes, commit that byte
//!    and repeat.  Stop if EOS is allowed or bytes diverge.
//!
//! 2. **Greedy tokenize with stop** — greedily tokenize the forced byte prefix
//!    from left to right, choosing the longest matching vocab token at each
//!    position.  Stop if a longer token could extend beyond the forced prefix
//!    (because we cannot determine the true longest match).

use crate::runtime::state::ConstraintState;

impl<'a> ConstraintState<'a> {
    /// Compute the forced token sequence.
    ///
    /// Uses two strategies, mirroring sep1's behavior:
    ///
    /// 1. **Byte-level** — find the longest byte prefix where all token first
    ///    bytes agree, then greedy tokenize.  Handles cases where multiple
    ///    tokens share a prefix (e.g. `"h"` and `"hello"`).
    ///
    /// 2. **Single-token fallback** — if exactly one token is allowed in the
    ///    mask, force it directly.  Handles multibyte tokens that span
    ///    multiple grammar terminals (e.g. `"ab"` matching `'a' 'b'`).
    pub fn force(&self) -> Vec<u32> {
        // If the parse is already complete, nothing to force.
        if self.is_complete() {
            return Vec::new();
        }

        // Try byte-level first.
        let forced_bytes = self.compute_forced_byte_prefix();
        let byte_level = self.tokenize_forced_with_stop(&forced_bytes);
        if !byte_level.is_empty() {
            return byte_level;
        }

        // Fall back to single-token force.
        self.single_token_force()
    }

    /// Single-token force: if there's exactly one allowed token at each step,
    /// commit it.  Repeats until ambiguity or completion.
    fn single_token_force(&self) -> Vec<u32> {
        let mut forced = Vec::new();
        let mut cursor = self.clone();

        loop {
            let mask = cursor.mask();
            let Some(token) = single_allowed_token(&mask) else {
                break;
            };
            forced.push(token);
            cursor.commit_token(token);
            if cursor.state.is_empty() || cursor.is_complete() {
                break;
            }
        }

        forced
    }

    /// Find the longest byte prefix where all allowed tokens agree on the
    /// first byte at each position.
    fn compute_forced_byte_prefix(&self) -> Vec<u8> {
        let token_bytes = &self.constraint.token_bytes;
        let eos = self.constraint.eos_token_id;
        let mut bytes = Vec::new();
        let mut cursor = self.clone();
        const MAX_FORCED_BYTES: usize = 10_000;

        loop {
            if bytes.len() >= MAX_FORCED_BYTES {
                break;
            }

            let mask = cursor.mask();

            // Check EOS: if the parse can complete here, stop forcing.
            if let Some(eos_id) = eos {
                if is_token_set(&mask, eos_id) {
                    break;
                }
            }

            // Find the unique first byte across all allowed tokens.
            let mut forced_byte: Option<u8> = None;
            let mut ambiguous = false;
            let mut any_token = false;

            for_each_set_bit(&mask, |token_id| {
                let Some(tb) = token_bytes.get(&token_id) else {
                    return; // token has no byte representation, skip
                };
                if tb.is_empty() {
                    return; // skip empty tokens
                }
                any_token = true;
                let fb = tb[0];
                match forced_byte {
                    None => forced_byte = Some(fb),
                    Some(prev) if prev == fb => {} // same
                    _ => ambiguous = true,
                }
            });

            if ambiguous || !any_token {
                break;
            }

            match forced_byte {
                Some(b) => {
                    bytes.push(b);
                    cursor.commit_bytes(&[b]);
                    if cursor.state.is_empty() {
                        bytes.pop(); // last byte killed the state
                        break;
                    }
                }
                None => break,
            }
        }

        bytes
    }

    /// Greedy left-to-right tokenization of forced bytes, stopping when the
    /// tokenizer would need to look beyond the forced prefix to determine the
    /// longest match.
    fn tokenize_forced_with_stop(&self, forced_bytes: &[u8]) -> Vec<u32> {
        let token_bytes = &self.constraint.token_bytes;
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < forced_bytes.len() {
            let remaining = &forced_bytes[pos..];
            let mut best_match: Option<(u32, usize)> = None;
            let mut could_extend_beyond = false;

            for (&token_id, tb) in token_bytes {
                if tb.is_empty() {
                    continue;
                }
                if remaining.starts_with(tb) {
                    // Full match at this position.
                    match best_match {
                        Some((_, prev_len)) if tb.len() <= prev_len => {}
                        _ => best_match = Some((token_id, tb.len())),
                    }
                } else if tb.starts_with(remaining) && tb.len() > remaining.len() {
                    // Token extends beyond the forced prefix — stop.
                    could_extend_beyond = true;
                }
            }

            if could_extend_beyond {
                // Cannot determine the true longest match; stop here.
                break;
            }

            match best_match {
                Some((token_id, len)) => {
                    tokens.push(token_id);
                    pos += len;
                }
                None => break, // No matching token at this position
            }
        }

        tokens
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a token ID is set in the bitmask.
fn is_token_set(mask: &[u32], token_id: u32) -> bool {
    let word_index = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word_index).map_or(false, |w| w & (1 << bit) != 0)
}

/// Return the single allowed token in the mask, or None if 0 or 2+ tokens.
fn single_allowed_token(mask: &[u32]) -> Option<u32> {
    let mut found = None;
    for (word_index, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as u32;
            let token = word_index as u32 * 32 + bit;
            if found.replace(token).is_some() {
                return None;
            }
            bits &= bits - 1;
        }
    }
    found
}

/// Iterate over all set bits in the bitmask, calling `f(token_id)` for each.
fn for_each_set_bit(mask: &[u32], mut f: impl FnMut(u32)) {
    for (word_index, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as u32;
            let token_id = word_index as u32 * 32 + bit;
            f(token_id);
            bits &= bits - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn test_forced_single_token() {
        // Only token "a" (id=0) is valid — should be forced.
        let vocab = make_vocab(&["a", "b"]);
        let c = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
        let s = c.start();
        let forced = s.force();
        assert_eq!(forced, vec![0]);
    }

    #[test]
    fn test_forced_shared_prefix_byte_level() {
        // Grammar: both alternatives share prefix "ab".
        // Vocab: "a"(0), "b"(1), "c"(2), "d"(3), "ab"(4).
        //
        // Forced bytes: "ab" (2 bytes).
        // "ab" (token 4) is exactly 2 bytes → fits within the forced prefix.
        // Greedy picks "ab" (len 2) over "a" (len 1).
        // Result: [4] = ["ab"].
        let vocab = make_vocab(&["a", "b", "c", "d", "ab"]);
        let c = Constraint::from_ebnf(
            r#"start ::= X_C | X_D
               X_C ::= "a" "b" "c"
               X_D ::= "a" "b" "d""#,
            &vocab,
        )
        .unwrap();
        let s = c.start();
        let forced = s.force();
        // "a" and "b" bytes are forced (shared prefix). Greedy tokenize: "ab"(4).
        assert_eq!(forced, vec![4], "greedy tokenize of forced prefix 'ab' should yield token 4");
    }

    #[test]
    fn test_forced_no_force_ambiguous() {
        // Two tokens "a" and "b" — different first bytes.
        let vocab = make_vocab(&["a", "b"]);
        let c = Constraint::from_ebnf(
            r#"start ::= "a" | "b""#,
            &vocab,
        )
        .unwrap();
        let s = c.start();
        let forced = s.force();
        assert!(forced.is_empty(), "ambiguous first byte, nothing forced");
    }

    #[test]
    fn test_forced_empty() {
        let mask = vec![0u32; 1];
        assert!(!is_token_set(&mask, 0));
    }

    #[test]
    fn test_is_token_set() {
        let mut mask = vec![0u32; 2];
        mask[0] |= 1u32 << 5;
        mask[1] |= 1u32 << 3; // token 35
        assert!(is_token_set(&mask, 5));
        assert!(is_token_set(&mask, 35));
        assert!(!is_token_set(&mask, 0));
        assert!(!is_token_set(&mask, 6));
    }
}
