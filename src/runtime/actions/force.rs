//! Forced-token computation.
//!
//! Uses the current `force()` algorithm:
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

enum ForcedFirstByte {
    None,
    Unique(u8),
    Ambiguous,
}

enum GreedyTokenizationStep {
    Match { token_id: u32, width: usize },
    BlockedByLongerToken,
    NoMatch,
}

impl<'a> ConstraintState<'a> {
    /// Compute the forced token sequence.
    ///
    /// Uses two strategies that preserve the established behavior:
    ///
    /// 1. **Byte-level** — find the longest byte prefix where all token first
    ///    bytes agree, then greedy tokenize.  Handles cases where multiple
    ///    tokens share a prefix (e.g. `"h"` and `"hello"`).
    ///
    /// 2. **Single-token fallback** — if exactly one token is allowed in the
    ///    mask, force it directly.  Handles multibyte tokens that span
    ///    multiple grammar terminals (e.g. `"ab"` matching `'a' 'b'`).
    pub fn force(&self) -> Vec<u32> {
        if self.is_complete() {
            return Vec::new();
        }

        self.force_by_bytes()
            .unwrap_or_else(|| self.single_token_force())
    }

    fn force_by_bytes(&self) -> Option<Vec<u32>> {
        let forced_bytes = self.compute_forced_byte_prefix();
        let tokens = self.tokenize_forced_with_stop(&forced_bytes);
        (!tokens.is_empty()).then_some(tokens)
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
            cursor.commit_token(token).expect("forced token should be in vocabulary");
            if cursor.state.is_empty() || cursor.is_complete() {
                break;
            }
        }

        forced
    }

    /// Find the longest byte prefix where all allowed tokens agree on the
    /// first byte at each position.
    fn compute_forced_byte_prefix(&self) -> Vec<u8> {
        let eos = self.constraint.eos_token_id;
        let mut bytes = Vec::new();
        let mut cursor = self.clone();
        const MAX_FORCED_BYTES: usize = 10_000;

        loop {
            if bytes.len() >= MAX_FORCED_BYTES {
                break;
            }

            let mask = cursor.mask();
            if let Some(eos_id) = eos {
                if is_token_set(&mask, eos_id) {
                    break;
                }
            }

            match cursor.forced_first_byte(&mask) {
                ForcedFirstByte::Unique(byte) => {
                    bytes.push(byte);
                    let _ = cursor.commit_bytes(&[byte]);
                    if cursor.state.is_empty() {
                        bytes.pop();
                        break;
                    }
                }
                ForcedFirstByte::None | ForcedFirstByte::Ambiguous => break,
            }
        }

        bytes
    }

    fn forced_first_byte(&self, mask: &[u32]) -> ForcedFirstByte {
        let mut first_byte = None;
        let mut ambiguous = false;
        let mut saw_token = false;

        for_each_set_bit(mask, |token_id| {
            let Some(token_bytes) = self.constraint.token_bytes.get(&token_id) else {
                return;
            };
            let Some(byte) = token_bytes.first().copied() else {
                return;
            };

            saw_token = true;
            match first_byte {
                None => first_byte = Some(byte),
                Some(existing) if existing == byte => {}
                Some(_) => ambiguous = true,
            }
        });

        if !saw_token {
            ForcedFirstByte::None
        } else if ambiguous {
            ForcedFirstByte::Ambiguous
        } else {
            ForcedFirstByte::Unique(first_byte.expect("saw_token implies a first byte"))
        }
    }

    /// Greedy left-to-right tokenization of forced bytes, stopping when the
    /// tokenizer would need to look beyond the forced prefix to determine the
    /// longest match.
    fn tokenize_forced_with_stop(&self, forced_bytes: &[u8]) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < forced_bytes.len() {
            match self.greedy_tokenization_step(&forced_bytes[pos..]) {
                GreedyTokenizationStep::Match { token_id, width } => {
                    tokens.push(token_id);
                    pos += width;
                }
                GreedyTokenizationStep::BlockedByLongerToken
                | GreedyTokenizationStep::NoMatch => break,
            }
        }

        tokens
    }

    fn greedy_tokenization_step(&self, remaining: &[u8]) -> GreedyTokenizationStep {
        let mut best_match = None;
        let mut blocked_by_longer_token = false;

        for (&token_id, token_bytes) in self.constraint.token_bytes.iter() {
            if token_bytes.is_empty() {
                continue;
            }
            if remaining.starts_with(token_bytes) {
                match best_match {
                    Some((_, best_width)) if token_bytes.len() <= best_width => {}
                    _ => best_match = Some((token_id, token_bytes.len())),
                }
                continue;
            }
            if token_bytes.starts_with(remaining) && token_bytes.len() > remaining.len() {
                blocked_by_longer_token = true;
            }
        }

        if blocked_by_longer_token {
            GreedyTokenizationStep::BlockedByLongerToken
        } else if let Some((token_id, width)) = best_match {
            GreedyTokenizationStep::Match { token_id, width }
        } else {
            GreedyTokenizationStep::NoMatch
        }
    }
}

// Helpers.

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
