//! Saved 2026-10-19 public-pipeline false-rejection regressions,
//! excluding seed 3229 (historical separately restricted stall investigation).
//! The other eight seed grammars had 19 historical baseline false rejections;
//! this checks each at all three optimization settings and all four saved
//! feed styles: whole bytes, individual bytes, tokens, and paired tokens.
//! Frozen archive SHA256 1c3d4ba6453af11d0f39ea16029105a0a4d159d05e20956ef97153489deee9e3.

use glrmask::{BuildOptions, Grammar, Optimization, Vocab};

fn check_positive_word(lark: &str, bytes: &[u8], seed: u32) {
    let vocab = Vocab::new(vec![
        (0, vec![b'a']), (1, vec![b'b']), (2, vec![b'c']),
        (3, b"ab".to_vec()), (4, b"bc".to_vec()), (5, b"ca".to_vec()),
    ]);
    for mode in [Optimization::FastBuild, Optimization::Balanced, Optimization::FastRuntime] {
        let c = Grammar::from_lark(lark)
            .compile_with(&vocab, BuildOptions::default().optimization(mode))
            .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?}: {e:?}"));
        let mut state = c.start();
        assert!(state.commit_bytes(bytes).is_ok(), "seed {seed} mode {mode:?} falsely rejected {bytes:?}");
        assert!(state.is_accepting(), "seed {seed} mode {mode:?} did not accept {bytes:?}");
        // The historical source harness also checked incremental byte and
        // token feeds. Exercise those separately from whole-string commitment.
        let mut byte_state = c.start();
        for &byte in bytes {
            assert!(byte_state.commit_bytes(&[byte]).is_ok(), "seed {seed} mode {mode:?} byte {byte:?}");
        }
        assert!(byte_state.is_accepting(), "seed {seed} mode {mode:?} byte-by-byte acceptance");
        let mut token_state = c.start();
        for &byte in bytes {
            let token = u32::from(byte - b'a');
            assert!((token_state.mask()[0] & (1u32 << token)) != 0,
                "seed {seed} mode {mode:?}: expected token {token} absent from mask");
            assert!(token_state.commit_token(token).is_ok(),
                "seed {seed} mode {mode:?}: token {token} falsely rejected");
        }
        assert!(token_state.is_accepting(), "seed {seed} mode {mode:?}: token acceptance");
        // The original E2E harness packs the three cross-terminal byte pairs
        // into a single model token when the next two bytes match.
        let mut pair_state = c.start();
        let mut at = 0usize;
        while at < bytes.len() {
            let pair_token = bytes.get(at..at + 2).and_then(|pair| match pair {
                b"ab" => Some(3u32),
                b"bc" => Some(4u32),
                b"ca" => Some(5u32),
                _ => None,
            });
            let (token, consumed) = pair_token.map_or((u32::from(bytes[at] - b'a'), 1), |token| (token, 2));
            assert!((pair_state.mask()[0] & (1u32 << token)) != 0,
                "seed {seed} mode {mode:?}: expected paired token {token} absent from mask");
            assert!(pair_state.commit_token(token).is_ok(),
                "seed {seed} mode {mode:?}: paired token {token} falsely rejected");
            at += consumed;
        }
        assert!(pair_state.is_accepting(), "seed {seed} mode {mode:?}: paired-token acceptance");

        // Independently computed finite fixed-point CFG languages from the
        // ORIGINAL archived integer-symbol rules, before Lark lowering and
        // before any GLRMask normalization (terminals -1/-2/-3 = a/b/c).
        // Check all 364 possible complete words over a/b/c up to length 5,
        // both accepting and rejecting, for each of all 8 archived seeds.
        let bounded_oracle: Option<&[&str]> = match seed {
            // Archived integer-rule input sha256: 1a035c81ad06504510a5fbe555eebf92d80ca1f0aaae0d493693cb25d0bd4192
            1313 => Some(&["a", "ba", "ca", "bac", "bba", "bca", "cca", "acca", "baac", "bbac", "bbba", "bbca", "bcac", "bcca", "ccca", "accca", "baacc", "babac", "bacac", "bacca", "bbaac", "bbacc", "bbbac", "bbbba", "bbbca", "bbcac", "bbcca", "bcaac", "bccac", "bccca", "cacca"]),
            // Archived integer-rule input sha256: b61aed16b2d4f697e11a5fb87f517115d631242bbb5bd59965990f8534541885
            1730 => Some(&["", "b", "c", "bc", "cc", "bbc", "bcc", "cbc", "ccc", "bbbc", "bbcc", "bcbc", "cbbc", "cbcc", "ccbc", "bbbbc", "bbcbc", "bcbbc", "bcbcc", "bccbc", "cbbbc", "cbcbc", "ccbbc", "ccbcc", "cccbc"]),
            // Archived integer-rule input sha256: 0cdfe65104fd9c6ece8bd29522f5051fc0373a93aa6d4b483b335556e935d4a9
            2009 => Some(&["b", "c", "ab", "cb", "aab", "aba", "aca", "acb", "ccb", "aaab", "aaba", "aacb", "acba", "accb", "cacb", "cccb", "aaaba", "aaacb", "aabaa", "aacaa", "aacba", "aaccb", "acacb", "accba", "acccb", "caacb", "caccb", "ccacb", "ccccb"]),
            // Archived integer-rule input sha256: f09f3dcfadd8e3168bfd150c45d46275faf8036f2a3dab803749af62ae57eb32
            2374 => Some(&["", "b", "c", "ac", "ca", "cc", "abc", "acc", "bca", "cca", "aacc", "acac", "acca", "accc", "caca", "ccca", "aabcc", "aaccc", "abcac", "abcca", "accac", "accca", "bcaca", "ccaca"]),
            // Archived integer-rule input sha256: 00e3e8ae53799835a81e4c04c2fd347e93961d88f792da5660b2303765613d43
            2556 => Some(&["", "a", "b", "ab", "ba", "bb", "aba", "abb", "bab", "bbb", "aabb", "abab", "abbb", "babb", "bbab", "bbbb", "aabbb", "ababb", "abbab", "abbbb", "baabb", "babab", "babbb", "bbabb", "bbbbb"]),
            // Archived integer-rule input sha256: 0957805b2866074120b97c4a8cecedb8f9f946edc27c4c8c8b7a51c4da203a24
            3204 => Some(&["a", "c", "ab", "ca", "cc", "cab", "cca", "ccb", "ccc", "bacc", "cabb", "cbac", "ccab", "ccca", "cccb", "cccc", "babac", "cbacc", "ccabb", "ccbac", "cccab", "cccbb", "cccca", "ccccb", "ccccc"]),
            // Archived integer-rule input sha256: 8f8bd6f50406cb4ed3deafdff9815662475462a3808228b98d51a6f716779284
            3274 => Some(&["", "a", "aa", "ab", "ac", "bb", "aac", "abb", "bba", "aaac", "aabc", "aacc", "abbc", "bbbb", "aaacc", "aabbc", "abbac"]),
            // Archived integer-rule input sha256: b59189b0f28b7e0770ee64dc8abcc63d389b393fc6e8b4446f1fac0207cb4967
            3810 => Some(&["b", "c", "ca", "cc", "caa", "cac", "cba", "cbb", "cca", "ccb", "ccc", "caaa", "caac", "caba", "caca", "cacc", "cbba", "cbbc", "ccaa", "ccab", "ccac", "ccba", "ccbc", "ccca", "cccb", "cccc", "caaba", "caaca", "cacaa", "cacba", "cacca", "cbaba", "cbaca", "cbbaa", "cbbac", "cbbba", "cbbca", "cbbcc", "ccaaa", "ccaab", "ccaba", "ccabc", "ccaca", "ccacb", "ccbaa", "ccbab", "ccbac", "ccbba", "ccbbb", "ccbca", "ccbcc", "cccaa", "cccab", "cccba", "cccbb", "cccbc", "cccca", "ccccb"]),
            _ => None,
        };
        if let Some(expected) = bounded_oracle {
            let reloaded = glrmask::Constraint::load(c.save())
                .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?}: reload failed: {e:?}"));
            assert_eq!(c.start().mask(), reloaded.start().mask(),
                "seed {seed} mode {mode:?}: initial masks differ after save/load");
            for c in [&c, &reloaded] {
                for len in 0..=5usize {
                for number in 0..3usize.pow(len as u32) {
                    let mut word = vec![b'a'; len];
                    let mut value = number;
                    for byte in word.iter_mut().rev() {
                        *byte += (value % 3) as u8;
                        value /= 3;
                    }
                    let accepted_by_original_cfg = expected.iter().any(|x| x.as_bytes() == word);
                    let mut query = c.start();
                    let observed = query.commit_bytes(&word).is_ok() && query.is_accepting();
                    assert_eq!(observed, accepted_by_original_cfg,
                        "seed {seed} mode {mode:?} unexpected recognition for {word:?}");
                    if accepted_by_original_cfg {
                        // A valid complete word has a viable next token at
                        // every proper prefix, not just at the saved witness.
                        let mut incremental = c.start();
                        for &next_byte in &word {
                            let token = u32::from(next_byte - b'a');
                            assert_ne!(incremental.mask()[0] & (1u32 << token), 0,
                                "seed {seed} mode {mode:?} missing valid token {token} after a prefix of {word:?}");
                            incremental.commit_token(token).unwrap_or_else(|e| {
                                panic!("seed {seed} mode {mode:?} invalid token commit on {word:?}: {e:?}")
                            });
                        }
                        assert!(incremental.is_accepting(),
                            "seed {seed} mode {mode:?} token-path completion missing for {word:?}");
                    }
                }
                }
            }
        }
    }
}

#[test]
fn archived_public_acceptance_seed_1313() {
    // Historical baseline mismatch modes: balanced, fastruntime.
    let lark = r####"start: n2
n0: "c" | n1 n1
n1: "c" |  | "a" "c" "c" | n2 "c"
n2: "a" | n0 "c" "a" | "b" n2 n1
"####;
    check_positive_word(lark, b"bac", 1313);
}

#[test]
fn archived_public_acceptance_seed_1730() {
    // Historical baseline mismatch modes: fastbuild.
    let lark = r####"start: n2
n0: "c" | n2 "b" |
n1: "c" | "a"
n2: "b" |  | n0 n0 "c" | "b"
"####;
    check_positive_word(lark, b"bcc", 1730);
}

#[test]
fn archived_public_acceptance_seed_2009() {
    // Historical baseline mismatch modes: balanced, fastbuild, fastruntime.
    let lark = r####"start: n3
n0: "a" | "a" | n2 "a" | n1 n1
n1: "c" | n0 "c" |
n2: "a" |  | "a" "a"
n3: "c" | n0 "b" | "a" n3 "a"
"####;
    check_positive_word(lark, b"aca", 2009);
}

#[test]
fn archived_public_acceptance_seed_2374() {
    // Historical baseline mismatch modes: balanced, fastbuild, fastruntime.
    let lark = r####"start: n0
n0: "b" | n3 "c" |  | n0 "c" "a"
n1: "c" | n0 | n1 n0 n2 | n0 "c"
n2: "c" |
n3: "c" |  | "a" n0
"####;
    check_positive_word(lark, b"acc", 2374);
}

#[test]
fn archived_public_acceptance_seed_2556() {
    // Historical baseline mismatch modes: balanced, fastruntime.
    let lark = r####"start: n2
n0: "b" | "a" "b" |  | n0
n1: "b" | n0 "b"
n2: "b" | n2 | "b" | n0 n3
n3: "a" | n2 n1 | n0
"####;
    check_positive_word(lark, b"aba", 2556);
}

#[test]
fn archived_public_acceptance_seed_3204() {
    // Historical baseline mismatch modes: balanced, fastruntime.
    let lark = r####"start: n1
n0: "a" | "c" n1 | n0
n1: "c" | n0 | n0 "b" | n2 n2 "c"
n2: "c" | "b" "a" | "b" "a"
"####;
    check_positive_word(lark, b"ccb", 3204);
}

#[test]
fn archived_public_acceptance_seed_3274() {
    // Historical baseline mismatch modes: balanced, fastbuild, fastruntime.
    let lark = r####"start: n1
n0: "a" |  | "b" "b"
n1: "a" | "a" "b" | n0 n0 | "a" n1 "c"
"####;
    check_positive_word(lark, b"aac", 3274);
}

#[test]
fn archived_public_acceptance_seed_3810() {
    // Historical baseline mismatch modes: balanced, fastbuild, fastruntime.
    let lark = r####"start: n0
n0: "b" | "c" n3
n1: "a" | n2 n2 |  | n0 "b" n2
n2: "c" | "a" |
n3: "c" |  | n3 n0 "a" | n1 n2
"####;
    check_positive_word(lark, b"cccb", 3810);
}
