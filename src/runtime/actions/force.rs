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

        for (&token_id, token_bytes) in &self.constraint.token_bytes {
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

    fn make_vocab_with_ids(entries: &[(u32, &str)]) -> Vocab {
        let entries = entries
            .iter()
            .map(|(token_id, bytes)| (*token_id, bytes.as_bytes().to_vec()))
            .collect();
        Vocab::new(entries, None)
    }

    fn build_tokenize_test_constraint(vocab: &Vocab) -> Constraint {
        let mut entries = vocab.entries.clone();
        let dummy_id = entries.keys().next_back().copied().unwrap_or(0) + 1;
        entries.insert(dummy_id, b"<dummy>".to_vec());
        let augmented = Vocab {
            entries,
            eos_token_id: None,
        };
        Constraint::from_ebnf(r#"start ::= "<dummy>""#, &augmented).unwrap()
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

    // ---------------------------------------------------------------------
    // Additional force() regression tests
    // ---------------------------------------------------------------------

    #[test]
    fn test_force_greedy_picks_longest() {
        let vocab = make_vocab(&["a", "b", "c", "ab", "abc"]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= ABC
            ABC ::= 'a' 'b' 'c'
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![4], "greedy tokenization should pick 'abc'");
    }

    #[test]
    fn test_force_greedy_picks_cdef() {
        let vocab = make_vocab(&["a", "b", "c", "d", "e", "f", "ab", "cdef"]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= ABCDEF
            ABCDEF ::= 'a' 'b' 'c' 'd' 'e' 'f'
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![6, 7], "greedy tokenization should pick 'ab' + 'cdef'");
    }

    #[test]
    fn test_force_steve_steven_with_eos() {
        let vocab = make_vocab_with_ids(&[
            (0, "s"),
            (1, "t"),
            (2, "e"),
            (3, "v"),
            (4, "n"),
            (5, "ste"),
            (6, "ve"),
            (7, "ven"),
            (8, "<|endoftext|>"),
        ]);
        let c = Constraint::from_lark(
            r#"
            start: STEVE | STEVEN
            STEVE: "steve"
            STEVEN: "steven"
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![5], "EOS should stop forcing at the unsafe 've'/'ven' boundary");
    }

    #[test]
    fn test_force_steve_steven_no_eos() {
        let vocab = make_vocab_with_ids(&[
            (0, "s"),
            (1, "t"),
            (2, "e"),
            (3, "v"),
            (4, "n"),
            (5, "ste"),
            (6, "ve"),
            (7, "ven"),
        ]);
        let c = Constraint::from_lark(
            r#"
            start: STEVE | STEVEN
            STEVE: "steve"
            STEVEN: "steven"
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![5, 7], "without EOS the remaining forced bytes should tokenize as 'ste' + 'ven'");
    }

    #[test]
    fn test_force_cutoff_realistic_grammar() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "e"),
            (5, "f"),
            (6, "ab"),
            (7, "cde"),
            (8, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= ABC | ABCDEF
            ABC ::= 'a' 'b' 'c'
            ABCDEF ::= 'a' 'b' 'c' 'd' 'e' 'f'
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![6], "'cde' could extend beyond the forced prefix, so only 'ab' is safe");
    }

    #[test]
    fn test_force_cutoff_blocks_everything() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "abc"),
            (5, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= AB | ABCD
            AB ::= 'a' 'b'
            ABCD ::= 'a' 'b' 'c' 'd'
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert!(forced.is_empty(), "'abc' could extend beyond 'ab', so nothing is safely forceable");
    }

    #[test]
    fn test_force_eos_stops_at_optional_continuation() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= AB | ABC
            AB ::= 'a' 'b'
            ABC ::= 'a' 'b' 'c'
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![0, 1], "EOS should stop forcing at the shared optional continuation boundary");
    }

    #[test]
    fn test_force_multi_byte_same_first_byte() {
        let vocab = make_vocab(&["a", "b", "ab"]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= AB
            AB ::= 'a' 'b'
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![2], "shared first byte forcing should still greedily choose 'ab'");
    }

    #[test]
    fn test_force_repetition_with_eos() {
        let vocab = make_vocab_with_ids(&[(0, "a"), (1, "<|endoftext|>")]);
        let c = Constraint::from_lark(
            r#"
            start: A_PLUS
            A_PLUS: /a+/
            "#,
            &vocab,
        )
        .unwrap();

        let forced = c.start().force();
        assert_eq!(forced, vec![0], "the first 'a' must be forced before EOS becomes available");
    }

    #[test]
    fn test_eos_detection_after_byte_commits() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "<|endoftext|>"),
        ]);
        let c = Constraint::from_lark(
            r#"
            start: AB | ABC
            AB: "ab"
            ABC: "abc"
            "#,
            &vocab,
        )
        .unwrap();

        let mut s = c.start();
        s.commit_bytes(b"a").unwrap();
        assert!(!s.is_complete(), "after only 'a' the parse should not be complete");

        s.commit_bytes(b"b").unwrap();
        let mask = s.mask();
        assert!(s.is_complete(), "after byte-committing 'ab' the shorter branch should be complete");
        assert!(is_token_set(&mask, 3), "EOS must be visible in the mask after completing 'ab' byte by byte");
    }

    // ---------------------------------------------------------------------
    // compute_forced_byte_prefix regression tests
    // ---------------------------------------------------------------------

    #[test]
    fn test_cfbp_deterministic_string() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "ab"),
            (4, "abc"),
            (5, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= ABC
            ABC ::= 'a' 'b' 'c'
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"abc", "deterministic grammar should force the whole byte string");
    }

    #[test]
    fn test_cfbp_ambiguous_from_start() {
        let vocab = make_vocab_with_ids(&[(0, "a"), (1, "b"), (2, "<|endoftext|>")]);
        let c = Constraint::from_ebnf(r#"start ::= 'a' | 'b'"#, &vocab).unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert!(prefix.is_empty(), "different first bytes should yield no forced prefix");
    }

    #[test]
    fn test_cfbp_shared_prefix_then_branch() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= ABC | ABD
            ABC ::= 'a' 'b' 'c'
            ABD ::= 'a' 'b' 'd'
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"ab", "only the shared prefix should be forced");
    }

    #[test]
    fn test_cfbp_eos_stops_forcing() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "<|endoftext|>"),
        ]);
        let c = Constraint::from_lark(
            r#"
            start: AB | ABC
            AB: "ab"
            ABC: "abc"
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"ab", "EOS should stop byte forcing once the shorter branch can finish");
    }

    #[test]
    fn test_cfbp_steve_steven_with_eos() {
        let vocab = make_vocab_with_ids(&[
            (0, "s"),
            (1, "t"),
            (2, "e"),
            (3, "v"),
            (4, "n"),
            (5, "ste"),
            (6, "ve"),
            (7, "ven"),
            (8, "<|endoftext|>"),
        ]);
        let c = Constraint::from_lark(
            r#"
            start: STEVE | STEVEN
            STEVE: "steve"
            STEVEN: "steven"
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"steve", "with EOS present only the shared complete prefix should be forced");
    }

    #[test]
    fn test_cfbp_steve_steven_no_eos() {
        let vocab = make_vocab_with_ids(&[
            (0, "s"),
            (1, "t"),
            (2, "e"),
            (3, "v"),
            (4, "n"),
            (5, "ste"),
            (6, "ve"),
            (7, "ven"),
        ]);
        let c = Constraint::from_lark(
            r#"
            start: STEVE | STEVEN
            STEVE: "steve"
            STEVEN: "steven"
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"steven", "without EOS the longer continuation should remain fully forced");
    }

    #[test]
    fn test_cfbp_after_partial_commit() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "e"),
            (5, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= ABCDE
            ABCDE ::= 'a' 'b' 'c' 'd' 'e'
            "#,
            &vocab,
        )
        .unwrap();

        let mut s = c.start();
        s.commit_bytes(b"ab").unwrap();
        let prefix = s.compute_forced_byte_prefix();
        assert_eq!(prefix, b"cde", "after committing 'ab' the remaining suffix should still be forced");
    }

    #[test]
    fn test_cfbp_empty_when_complete() {
        let vocab = make_vocab_with_ids(&[(0, "a"), (1, "b"), (2, "<|endoftext|>")]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= AB
            AB ::= 'a' 'b'
            "#,
            &vocab,
        )
        .unwrap();

        let mut s = c.start();
        s.commit_bytes(b"ab").unwrap();
        let prefix = s.compute_forced_byte_prefix();
        assert!(prefix.is_empty(), "complete parses should have no further forced prefix");
    }

    #[test]
    fn test_cfbp_long_deterministic() {
        let entries: Vec<(u32, Vec<u8>)> = b"abcdefghijklmnop"
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, vec![*b]))
            .chain(std::iter::once((16, b"<|endoftext|>".to_vec())))
            .collect();
        let vocab = Vocab::new(entries, None);
        let c = Constraint::from_ebnf(
            r#"
            start ::= LONG
            LONG ::= 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j' 'k' 'l' 'm' 'n' 'o' 'p'
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"abcdefghijklmnop", "long deterministic literals should force their full byte sequence");
    }

    #[test]
    fn test_cfbp_single_char_grammar() {
        let vocab = make_vocab_with_ids(&[(0, "x"), (1, "<|endoftext|>")]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= X
            X ::= 'x'
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"x", "single-character grammars should force that byte");
    }

    #[test]
    fn test_cfbp_repetition_no_eos() {
        let vocab = make_vocab_with_ids(&[(0, "a")]);
        let c = Constraint::from_lark(
            r#"
            start: AS
            AS: /a+/
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix.len(), 10_000, "repetition without EOS should stop at the safety cutoff");
        assert!(prefix.iter().all(|&b| b == b'a'), "the forced repetition prefix should contain only 'a' bytes");
    }

    #[test]
    fn test_cfbp_repetition_with_eos() {
        let vocab = make_vocab_with_ids(&[(0, "a"), (1, "<|endoftext|>")]);
        let c = Constraint::from_lark(
            r#"
            start: AS
            AS: /a+/
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"a", "with EOS present only the first repeated byte should be forced");
    }

    #[test]
    fn test_cfbp_abcdef_with_multibyte_tokens() {
        let vocab = make_vocab_with_ids(&[
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (4, "e"),
            (5, "f"),
            (6, "ab"),
            (7, "cdef"),
            (8, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= ABCDEF
            ABCDEF ::= 'a' 'b' 'c' 'd' 'e' 'f'
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"abcdef", "multibyte vocab entries should not interfere with byte-prefix forcing");
    }

    #[test]
    fn test_cfbp_alternation_shared_long_prefix() {
        let vocab = make_vocab_with_ids(&[
            (0, "h"),
            (1, "e"),
            (2, "l"),
            (3, "o"),
            (4, "a"),
            (5, "b"),
            (6, "hello"),
            (7, "<|endoftext|>"),
        ]);
        let c = Constraint::from_ebnf(
            r#"
            start ::= OPT1 | OPT2
            OPT1 ::= 'h' 'e' 'l' 'l' 'o' 'a'
            OPT2 ::= 'h' 'e' 'l' 'l' 'o' 'b'
            "#,
            &vocab,
        )
        .unwrap();

        let prefix = c.start().compute_forced_byte_prefix();
        assert_eq!(prefix, b"hello", "only the shared long prefix should be forced");
    }

    // ---------------------------------------------------------------------
    // tokenize_forced_with_stop regression tests
    // ---------------------------------------------------------------------

    #[test]
    fn test_twfs_single_byte_tokens() {
        let vocab = make_vocab(&["a", "b", "c", "d", "e", "f"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"abcdef");
        assert_eq!(tokens, vec![0, 1, 2, 3, 4, 5], "single-byte tokens should tokenize one byte at a time");
    }

    #[test]
    fn test_twfs_greedy_longest_match() {
        let vocab = make_vocab(&["a", "b", "c", "ab", "abc"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"abc");
        assert_eq!(tokens, vec![4], "tokenization should prefer the longest full match");
    }

    #[test]
    fn test_twfs_stop_when_token_extends_beyond() {
        let vocab = make_vocab(&["a", "b", "c", "cde", "ab"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"abc");
        assert_eq!(tokens, vec![4], "tokenization should stop once a longer token could extend beyond the prefix");
    }

    #[test]
    fn test_twfs_stop_at_first_position() {
        let vocab = make_vocab(&["a", "b", "abc"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"ab");
        assert!(tokens.is_empty(), "an extending token at position 0 should block all output");
    }

    #[test]
    fn test_twfs_multibyte_middle() {
        let vocab = make_vocab(&["a", "b", "c", "d", "e", "f", "ab", "cdef"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"abcdef");
        assert_eq!(tokens, vec![6, 7], "multibyte matches in the middle of the prefix should be chosen greedily");
    }

    #[test]
    fn test_twfs_exact_fit_no_extension() {
        let vocab = make_vocab(&["abc", "def"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"abcdef");
        assert_eq!(tokens, vec![0, 1], "exact full coverage should emit all matching tokens");
    }

    #[test]
    fn test_twfs_empty_input() {
        let vocab = make_vocab(&["a"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"");
        assert!(tokens.is_empty(), "empty input should tokenize to an empty sequence");
    }

    #[test]
    fn test_twfs_no_matching_token() {
        let vocab = make_vocab(&["x", "y"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"abc");
        assert!(tokens.is_empty(), "no matching token at the current position should stop tokenization");
    }

    #[test]
    fn test_twfs_partial_coverage() {
        let vocab = make_vocab(&["a", "b"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"abc");
        assert_eq!(tokens, vec![0, 1], "tokenization should stop after the last covered byte");
    }

    #[test]
    fn test_twfs_ven_extends_beyond() {
        let vocab = make_vocab(&["s", "t", "e", "v", "ste", "ve", "ven"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"steve");
        assert_eq!(tokens, vec![4], "'ven' extending beyond the prefix should stop tokenization after 'ste'");
    }

    #[test]
    fn test_twfs_steven_full_coverage() {
        let vocab = make_vocab(&["s", "t", "e", "v", "n", "ste", "ve", "ven"]);
        let c = build_tokenize_test_constraint(&vocab);

        let tokens = c.start().tokenize_forced_with_stop(b"steven");
        assert_eq!(tokens, vec![5, 7], "full coverage without extension should emit 'ste' + 'ven'");
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
