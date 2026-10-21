//! Saved 2026-10-19 public-pipeline false-rejection regressions,
//! excluding seed 3229 (historical separately restricted stall investigation).
//! The other eight seed grammars had 19 historical baseline false rejections;
//! this checks each at all three optimization settings and all four saved
//! feed styles: whole bytes, individual bytes, tokens, and paired tokens.
//! Frozen archive SHA256 1c3d4ba6453af11d0f39ea16029105a0a4d159d05e20956ef97153489deee9e3.

use glrmask::{CompileOptions, Grammar, MaskStrategy, Module, Vocab};

fn check_positive_word(lark: &str, bytes: &[u8], seed: u32) {
    let vocab = Vocab::new(vec![
        (0, vec![b'a']), (1, vec![b'b']), (2, vec![b'c']),
        (3, b"ab".to_vec()), (4, b"bc".to_vec()), (5, b"ca".to_vec()),
    ]);
    let grammar = Grammar::from_lark(lark);
    for mode in [MaskStrategy::Auto, MaskStrategy::ParserDwa, MaskStrategy::TokenTrie, MaskStrategy::TokenClassTrie] {
        let mut module = Module::new(&grammar);
        module.set_mask_strategy(mode).unwrap();
        let c = module
            .compile(&vocab)
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
        // Check all 3,280 possible complete words over a/b/c up to length 7,
        // both accepting and rejecting, for each of all 8 archived seeds.
        let bounded_oracle: Option<&[&str]> = match seed {
            // Frozen source SHA256 1a035c81ad06504510a5fbe555eebf92d80ca1f0aaae0d493693cb25d0bd4192
            1313 => Some(&["a", "ba", "ca", "bac", "bba", "bca", "cca", "acca", "baac", "bbac", "bbba", "bbca", "bcac", "bcca", "ccca", "accca", "baacc", "babac", "bacac", "bacca", "bbaac", "bbacc", "bbbac", "bbbba", "bbbca", "bbcac", "bbcca", "bcaac", "bccac", "bccca", "cacca", "acacca", "acccca", "babacc", "babbac", "babcac", "baccac", "baccca", "bbaacc", "bbabac", "bbacac", "bbacca", "bbbaac", "bbbacc", "bbbbac", "bbbbba", "bbbbca", "bbbcac", "bbbcca", "bbcaac", "bbcacc", "bbccac", "bbccca", "bcaacc", "bcabac", "bcacac", "bcacca", "bccaac", "bcccac", "caccca", "cbacca", "ccacca", "acaccca", "acbacca", "accacca", "baaccac", "baaccca", "babaacc", "babbacc", "babbbac", "babbcac", "babcacc", "babccac", "bacacca", "baccaac", "bacccac", "bacccca", "bbaacac", "bbaaccc", "bbabacc", "bbabbac", "bbabcac", "bbacacc", "bbacbac", "bbaccac", "bbaccca", "bbbaacc", "bbbabac", "bbbacac", "bbbacca", "bbbaccc", "bbbbaac", "bbbbacc", "bbbbbac", "bbbbbba", "bbbbbca", "bbbbcac", "bbbbcca", "bbbcaac", "bbbcacc", "bbbccac", "bbbccca", "bbcaacc", "bbcabac", "bbcacac", "bbcacca", "bbccaac", "bbccacc", "bbcccac", "bcabacc", "bcabbac", "bcabcac", "bcaccac", "bcaccca", "bcbacca", "bccaacc", "bccabac", "bccacac", "bccacca", "bcccaac", "cacacca", "cbaccca", "cbbacca", "cbcacca", "ccaccca", "cccacca"]),
            // Frozen source SHA256 b61aed16b2d4f697e11a5fb87f517115d631242bbb5bd59965990f8534541885
            1730 => Some(&["", "b", "c", "bc", "cc", "bbc", "bcc", "cbc", "ccc", "bbbc", "bbcc", "bcbc", "cbbc", "cbcc", "ccbc", "bbbbc", "bbcbc", "bcbbc", "bcbcc", "bccbc", "cbbbc", "cbcbc", "ccbbc", "ccbcc", "cccbc", "bbbcbc", "bbcbbc", "bbcbcc", "bbccbc", "bcbbbc", "bcbcbc", "bccbbc", "bccbcc", "bcccbc", "cbbcbc", "cbcbbc", "cbcbcc", "cbccbc", "ccbbbc", "ccbcbc", "cccbbc", "cccbcc", "ccccbc", "bbbbcbc", "bbbcbbc", "bbbcbcc", "bbbccbc", "bbcbbbc", "bbcbcbc", "bbccbbc", "bbccbcc", "bbcccbc", "bcbbcbc", "bcbcbbc", "bcbcbcc", "bcbccbc", "bccbbbc", "bccbcbc", "cbbbcbc", "cbbcbbc", "cbbcbcc", "cbbccbc", "cbcbbbc", "cbcbcbc", "cbccbbc", "cbccbcc", "cbcccbc", "ccbbcbc", "ccbcbbc", "ccbcbcc", "ccbccbc", "cccbbbc", "cccbcbc"]),
            // Frozen source SHA256 0cdfe65104fd9c6ece8bd29522f5051fc0373a93aa6d4b483b335556e935d4a9
            2009 => Some(&["b", "c", "ab", "cb", "aab", "aba", "aca", "acb", "ccb", "aaab", "aaba", "aacb", "acba", "accb", "cacb", "cccb", "aaaba", "aaacb", "aabaa", "aacaa", "aacba", "aaccb", "acacb", "accba", "acccb", "caacb", "caccb", "ccacb", "ccccb", "aaaaba", "aaabaa", "aaacba", "aaaccb", "aacacb", "aacbaa", "aaccba", "aacccb", "acaacb", "acacba", "acaccb", "accacb", "acccba", "accccb", "caaacb", "caaccb", "cacccb", "ccaacb", "ccaccb", "cccacb", "cccccb", "aaaabaa", "aaaacba", "aaabaaa", "aaacaaa", "aaacacb", "aaacbaa", "aaaccba", "aaacccb", "aacaacb", "aacacba", "aacaccb", "aaccacb", "aaccbaa", "aacccba", "aaccccb", "acaaacb", "acaacba", "acaaccb", "acaccba", "acacccb", "accaacb", "accacba", "accaccb", "acccacb", "accccba", "acccccb", "caaaccb", "caacccb", "cacaccb", "caccacb", "caccccb", "ccaaacb", "ccaaccb", "ccacccb", "cccaacb", "cccaccb", "ccccacb", "ccccccb"]),
            // Frozen source SHA256 f09f3dcfadd8e3168bfd150c45d46275faf8036f2a3dab803749af62ae57eb32
            2374 => Some(&["", "b", "c", "ac", "ca", "cc", "abc", "acc", "bca", "cca", "aacc", "acac", "acca", "accc", "caca", "ccca", "aabcc", "aaccc", "abcac", "abcca", "accac", "accca", "bcaca", "ccaca", "aaaccc", "aacacc", "aaccac", "aaccca", "aacccc", "acacac", "acacca", "accaca", "acccac", "acccca", "cacaca", "cccaca", "aaabccc", "aaacccc", "aabcacc", "aabccac", "aabccca", "aaccacc", "aacccac", "aacccca", "abcacac", "abcacca", "abccaca", "accacac", "accacca", "acccaca", "bcacaca", "ccacaca"]),
            // Frozen source SHA256 00e3e8ae53799835a81e4c04c2fd347e93961d88f792da5660b2303765613d43
            2556 => Some(&["", "a", "b", "ab", "ba", "bb", "aba", "abb", "bab", "bbb", "aabb", "abab", "abbb", "babb", "bbab", "bbbb", "aabbb", "ababb", "abbab", "abbbb", "baabb", "babab", "babbb", "bbabb", "bbbbb", "aabbbb", "abaabb", "ababab", "ababbb", "abbabb", "abbbbb", "baabbb", "bababb", "babbbb", "bbaabb", "bbabbb", "bbbabb", "bbbbbb", "aabbabb", "aabbbbb", "abaabbb", "abababb", "ababbbb", "abbaabb", "abbabbb", "abbbabb", "abbbbbb", "baabbbb", "babaabb", "bababbb", "babbabb", "babbbbb", "bbaabbb", "bbababb", "bbabbbb", "bbbabbb", "bbbbabb", "bbbbbbb"]),
            // Frozen source SHA256 0957805b2866074120b97c4a8cecedb8f9f946edc27c4c8c8b7a51c4da203a24
            3204 => Some(&["a", "c", "ab", "ca", "cc", "cab", "cca", "ccb", "ccc", "bacc", "cabb", "cbac", "ccab", "ccca", "cccb", "cccc", "babac", "cbacc", "ccabb", "ccbac", "cccab", "cccbb", "cccca", "ccccb", "ccccc", "cbabac", "cbaccb", "ccabbb", "ccbacb", "ccbacc", "cccabb", "cccbac", "ccccab", "ccccbb", "ccccca", "cccccb", "cccccc", "cbabacb", "ccbabac", "ccbaccb", "cccabbb", "cccbacb", "cccbacc", "ccccabb", "ccccbac", "ccccbbb", "cccccab", "cccccbb", "cccccca", "ccccccb", "ccccccc"]),
            // Frozen source SHA256 8f8bd6f50406cb4ed3deafdff9815662475462a3808228b98d51a6f716779284
            3274 => Some(&["", "a", "aa", "ab", "ac", "bb", "aac", "abb", "bba", "aaac", "aabc", "aacc", "abbc", "bbbb", "aaacc", "aabbc", "abbac", "aaaacc", "aaabcc", "aaaccc", "aabbcc", "abbbbc", "aaaaccc", "aaabbcc", "aabbacc"]),
            // Frozen source SHA256 b59189b0f28b7e0770ee64dc8abcc63d389b393fc6e8b4446f1fac0207cb4967
            3810 => Some(&["b", "c", "ca", "cc", "caa", "cac", "cba", "cbb", "cca", "ccb", "ccc", "caaa", "caac", "caba", "caca", "cacc", "cbba", "cbbc", "ccaa", "ccab", "ccac", "ccba", "ccbc", "ccca", "cccb", "cccc", "caaba", "caaca", "cacaa", "cacba", "cacca", "cbaba", "cbaca", "cbbaa", "cbbac", "cbbba", "cbbca", "cbbcc", "ccaaa", "ccaab", "ccaba", "ccabc", "ccaca", "ccacb", "ccbaa", "ccbab", "ccbac", "ccbba", "ccbbb", "ccbca", "ccbcc", "cccaa", "cccab", "cccba", "cccbb", "cccbc", "cccca", "ccccb", "caaaba", "caaaca", "caacaa", "caacba", "caacca", "cababa", "cabaca", "cacaaa", "cacaba", "cacaca", "cacbaa", "cacbba", "caccaa", "caccba", "caccca", "cbacaa", "cbacca", "cbbaba", "cbbaca", "cbbcaa", "cbbcba", "cbbcca", "ccaaaa", "ccaaab", "ccaaba", "ccaabc", "ccaaca", "ccaacb", "ccabaa", "ccabab", "ccabac", "ccabba", "ccabca", "ccabcc", "ccacaa", "ccacab", "ccacba", "ccacbc", "ccacca", "ccaccb", "ccbaba", "ccbabc", "ccbaca", "ccbbaa", "ccbbab", "ccbbba", "ccbbbc", "ccbbca", "ccbbcb", "ccbcaa", "ccbcba", "ccbcca", "cccaaa", "cccaab", "cccaba", "cccabb", "cccabc", "cccaca", "cccacb", "cccbaa", "cccbab", "cccbac", "cccbba", "cccbbc", "cccbca", "cccbcb", "cccbcc", "ccccaa", "ccccab", "ccccba", "ccccbb", "ccccbc", "ccccca", "cccccb", "caaacaa", "caaacca", "caababa", "caabaca", "caacaaa", "caacaba", "caacaca", "caacbaa", "caacbba", "caaccaa", "caaccba", "caaccca", "cabacaa", "cabacca", "cacaaaa", "cacaaba", "cacaaca", "cacabaa", "cacacaa", "cacacca", "cacbaba", "cacbaca", "cacbbaa", "cacbbca", "caccaaa", "caccaba", "caccaca", "caccbaa", "caccbba", "caccbca", "cacccaa", "cacccba", "cacccca", "cbababa", "cbabaca", "cbacaaa", "cbacaba", "cbacaca", "cbacbaa", "cbacbba", "cbaccaa", "cbaccba", "cbaccca", "cbbaaba", "cbbaaca", "cbbacaa", "cbbacba", "cbbacca", "cbbbaba", "cbbbaca", "cbbcaaa", "cbbcaba", "cbbcaca", "cbbcbaa", "cbbcbba", "cbbccaa", "cbbccba", "cbbccca", "ccaaaba", "ccaaabc", "ccaaaca", "ccaabaa", "ccaabab", "ccaabac", "ccaabba", "ccaabca", "ccaabcc", "ccaacaa", "ccaacab", "ccaacba", "ccaacbc", "ccaacca", "ccababa", "ccababc", "ccabaca", "ccabcaa", "ccabcba", "ccabcca", "ccacaaa", "ccacaab", "ccacaba", "ccacabc", "ccacaca", "ccacbaa", "ccacbab", "ccacbac", "ccacbba", "ccacbca", "ccacbcc", "ccaccaa", "ccaccab", "ccaccba", "ccaccbc", "ccaccca", "ccbaaba", "ccbaaca", "ccbabaa", "ccbabab", "ccbabac", "ccbabba", "ccbabca", "ccbabcc", "ccbacaa", "ccbacab", "ccbacba", "ccbacca", "ccbbaaa", "ccbbaab", "ccbbaba", "ccbbabc", "ccbbaca", "ccbbacb", "ccbbbaa", "ccbbbab", "ccbbbac", "ccbbbba", "ccbbbca", "ccbbbcc", "ccbbcaa", "ccbbcab", "ccbbcba", "ccbbcbc", "ccbbcca", "ccbbccb", "ccbcaaa", "ccbcaba", "ccbcaca", "ccbcbaa", "ccbcbba", "ccbccaa", "ccbccba", "ccbccca", "cccaaaa", "cccaaab", "cccaaba", "cccaabb", "cccaabc", "cccaaca", "cccabaa", "cccabab", "cccabac", "cccabba", "cccabbc", "cccabca", "cccabcb", "cccabcc", "cccacaa", "cccacab", "cccacba", "cccacbb", "cccacbc", "cccacca", "cccbaaa", "cccbaab", "cccbaba", "cccbabb", "cccbabc", "cccbaca", "cccbacb", "cccbbaa", "cccbbab", "cccbbac", "cccbbba", "cccbbbb", "cccbbca", "cccbbcc", "cccbcaa", "cccbcab", "cccbcba", "cccbcbc", "cccbcca", "cccbccb", "ccccaaa", "ccccaab", "ccccaba", "ccccabb", "ccccabc", "ccccaca", "ccccbaa", "ccccbab", "ccccbac", "ccccbba", "ccccbbb", "ccccbbc", "ccccbca", "ccccbcb", "ccccbcc", "cccccaa", "cccccab", "cccccba", "cccccbb", "cccccbc", "cccccca"]),
            _ => None,
        };
        // This oracle is independently derived from original integer CFG
        // productions using a bounded-prefix fixed point, including words
        // LONGER than the bound. Unlike truncating complete-word enumeration,
        // this distinguishes a dead prefix from a viable long continuation.
        // Includes all viable prefixes up to 5 bytes (including epsilon).
        let prefix_oracle: Option<&[&str]> = match seed {
            // Original input SHA256 1a035c81ad06504510a5fbe555eebf92d80ca1f0aaae0d493693cb25d0bd4192
            1313 => Some(&["", "a", "b", "c", "ac", "ba", "bb", "bc", "ca", "cb", "cc", "aca", "acb", "acc", "baa", "bab", "bac", "bba", "bbb", "bbc", "bca", "bcb", "bcc", "cac", "cba", "cbb", "cbc", "cca", "ccb", "ccc", "acac", "acba", "acbb", "acbc", "acca", "accb", "accc", "baac", "baba", "babb", "babc", "baca", "bacb", "bacc", "bbaa", "bbab", "bbac", "bbba", "bbbb", "bbbc", "bbca", "bbcb", "bbcc", "bcaa", "bcab", "bcac", "bcba", "bcbb", "bcbc", "bcca", "bccb", "bccc", "caca", "cacb", "cacc", "cbaa", "cbab", "cbac", "cbba", "cbbb", "cbbc", "cbca", "cbcb", "cbcc", "ccac", "ccba", "ccbb", "ccbc", "ccca", "cccb", "cccc", "acaca", "acacb", "acacc", "acbaa", "acbab", "acbac", "acbba", "acbbb", "acbbc", "acbca", "acbcb", "acbcc", "accac", "accba", "accbb", "accbc", "accca", "acccb", "acccc", "baaca", "baacb", "baacc", "babaa", "babab", "babac", "babba", "babbb", "babbc", "babca", "babcb", "babcc", "bacac", "bacba", "bacbb", "bacbc", "bacca", "baccb", "baccc", "bbaac", "bbaba", "bbabb", "bbabc", "bbaca", "bbacb", "bbacc", "bbbaa", "bbbab", "bbbac", "bbbba", "bbbbb", "bbbbc", "bbbca", "bbbcb", "bbbcc", "bbcaa", "bbcab", "bbcac", "bbcba", "bbcbb", "bbcbc", "bbcca", "bbccb", "bbccc", "bcaac", "bcaba", "bcabb", "bcabc", "bcaca", "bcacb", "bcacc", "bcbaa", "bcbab", "bcbac", "bcbba", "bcbbb", "bcbbc", "bcbca", "bcbcb", "bcbcc", "bccaa", "bccab", "bccac", "bccba", "bccbb", "bccbc", "bccca", "bcccb", "bcccc", "cacac", "cacba", "cacbb", "cacbc", "cacca", "caccb", "caccc", "cbaac", "cbaba", "cbabb", "cbabc", "cbaca", "cbacb", "cbacc", "cbbaa", "cbbab", "cbbac", "cbbba", "cbbbb", "cbbbc", "cbbca", "cbbcb", "cbbcc", "cbcaa", "cbcab", "cbcac", "cbcba", "cbcbb", "cbcbc", "cbcca", "cbccb", "cbccc", "ccaca", "ccacb", "ccacc", "ccbaa", "ccbab", "ccbac", "ccbba", "ccbbb", "ccbbc", "ccbca", "ccbcb", "ccbcc", "cccac", "cccba", "cccbb", "cccbc", "cccca", "ccccb", "ccccc"]),
            // Original input SHA256 b61aed16b2d4f697e11a5fb87f517115d631242bbb5bd59965990f8534541885
            1730 => Some(&["", "b", "c", "bb", "bc", "cb", "cc", "bbb", "bbc", "bcb", "bcc", "cbb", "cbc", "ccb", "ccc", "bbbb", "bbbc", "bbcb", "bbcc", "bcbb", "bcbc", "bccb", "bccc", "cbbb", "cbbc", "cbcb", "cbcc", "ccbb", "ccbc", "cccb", "cccc", "bbbbb", "bbbbc", "bbbcb", "bbbcc", "bbcbb", "bbcbc", "bbccb", "bbccc", "bcbbb", "bcbbc", "bcbcb", "bcbcc", "bccbb", "bccbc", "bcccb", "bcccc", "cbbbb", "cbbbc", "cbbcb", "cbbcc", "cbcbb", "cbcbc", "cbccb", "cbccc", "ccbbb", "ccbbc", "ccbcb", "ccbcc", "cccbb", "cccbc", "ccccb", "ccccc"]),
            // Original input SHA256 0cdfe65104fd9c6ece8bd29522f5051fc0373a93aa6d4b483b335556e935d4a9
            2009 => Some(&["", "a", "b", "c", "aa", "ab", "ac", "ca", "cb", "cc", "aaa", "aab", "aac", "aba", "aca", "acb", "acc", "caa", "cac", "cca", "ccb", "ccc", "aaaa", "aaab", "aaac", "aaba", "aaca", "aacb", "aacc", "acaa", "acac", "acba", "acca", "accb", "accc", "caaa", "caac", "caca", "cacb", "cacc", "ccaa", "ccac", "ccca", "cccb", "cccc", "aaaaa", "aaaab", "aaaac", "aaaba", "aaaca", "aaacb", "aaacc", "aabaa", "aacaa", "aacac", "aacba", "aacca", "aaccb", "aaccc", "acaaa", "acaac", "acaca", "acacb", "acacc", "accaa", "accac", "accba", "accca", "acccb", "acccc", "caaac", "caaca", "caacb", "caacc", "cacaa", "cacac", "cacca", "caccb", "caccc", "ccaaa", "ccaac", "ccaca", "ccacb", "ccacc", "cccaa", "cccac", "cccca", "ccccb", "ccccc"]),
            // Original input SHA256 f09f3dcfadd8e3168bfd150c45d46275faf8036f2a3dab803749af62ae57eb32
            2374 => Some(&["", "a", "b", "c", "aa", "ab", "ac", "bc", "ca", "cc", "aaa", "aab", "aac", "abc", "aca", "acc", "bca", "cac", "cca", "ccc", "aaaa", "aaab", "aaac", "aabc", "aaca", "aacc", "abca", "abcc", "acac", "acca", "accc", "bcac", "caca", "ccac", "ccca", "aaaaa", "aaaab", "aaaac", "aaabc", "aaaca", "aaacc", "aabca", "aabcc", "aacac", "aacca", "aaccc", "abcac", "abcca", "acaca", "acacc", "accac", "accca", "acccc", "bcaca", "cacac", "ccaca", "cccac"]),
            // Original input SHA256 00e3e8ae53799835a81e4c04c2fd347e93961d88f792da5660b2303765613d43
            2556 => Some(&["", "a", "b", "aa", "ab", "ba", "bb", "aab", "aba", "abb", "baa", "bab", "bba", "bbb", "aabb", "abaa", "abab", "abba", "abbb", "baab", "baba", "babb", "bbaa", "bbab", "bbba", "bbbb", "aabba", "aabbb", "abaab", "ababa", "ababb", "abbaa", "abbab", "abbba", "abbbb", "baabb", "babaa", "babab", "babba", "babbb", "bbaab", "bbaba", "bbabb", "bbbaa", "bbbab", "bbbba", "bbbbb"]),
            // Original input SHA256 0957805b2866074120b97c4a8cecedb8f9f946edc27c4c8c8b7a51c4da203a24
            3204 => Some(&["", "a", "b", "c", "ab", "ba", "ca", "cb", "cc", "bab", "bac", "cab", "cba", "cca", "ccb", "ccc", "baba", "bacc", "cabb", "cbab", "cbac", "ccab", "ccba", "ccca", "cccb", "cccc", "babac", "cbaba", "cbacc", "ccabb", "ccbab", "ccbac", "cccab", "cccba", "cccbb", "cccca", "ccccb", "ccccc"]),
            // Original input SHA256 8f8bd6f50406cb4ed3deafdff9815662475462a3808228b98d51a6f716779284
            3274 => Some(&["", "a", "b", "aa", "ab", "ac", "bb", "aaa", "aab", "aac", "abb", "bba", "bbb", "aaaa", "aaab", "aaac", "aabb", "aabc", "aacc", "abba", "abbb", "abbc", "bbbb", "aaaaa", "aaaab", "aaaac", "aaabb", "aaabc", "aaacc", "aabba", "aabbb", "aabbc", "abbac", "abbbb"]),
            // Original input SHA256 b59189b0f28b7e0770ee64dc8abcc63d389b393fc6e8b4446f1fac0207cb4967
            3810 => Some(&["", "b", "c", "ca", "cb", "cc", "caa", "cab", "cac", "cba", "cbb", "cca", "ccb", "ccc", "caaa", "caab", "caac", "caba", "caca", "cacb", "cacc", "cbab", "cbac", "cbba", "cbbb", "cbbc", "ccaa", "ccab", "ccac", "ccba", "ccbb", "ccbc", "ccca", "cccb", "cccc", "caaab", "caaac", "caaba", "caaca", "caacb", "caacc", "cabab", "cabac", "cacaa", "cacab", "cacac", "cacba", "cacbb", "cacca", "caccb", "caccc", "cbaba", "cbaca", "cbacb", "cbacc", "cbbaa", "cbbab", "cbbac", "cbbba", "cbbca", "cbbcb", "cbbcc", "ccaaa", "ccaab", "ccaac", "ccaba", "ccabb", "ccabc", "ccaca", "ccacb", "ccacc", "ccbaa", "ccbab", "ccbac", "ccbba", "ccbbb", "ccbbc", "ccbca", "ccbcb", "ccbcc", "cccaa", "cccab", "cccac", "cccba", "cccbb", "cccbc", "cccca", "ccccb", "ccccc"]),
            _ => None,
        };
        if let Some(expected) = bounded_oracle {
            let reloaded = glrmask::Constraint::load(c.save())
                .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?}: reload failed: {e:?}"));
            assert_eq!(c.start().mask(), reloaded.start().mask(),
                "seed {seed} mode {mode:?}: initial masks differ after save/load");
            for c in [&c, &reloaded] {
                let viable = prefix_oracle.expect("each tested seed needs independent prefix oracle");
                let token_bytes: [&[u8]; 6] = [b"a", b"b", b"c", b"ab", b"bc", b"ca"];
                for len in 0..=3usize {
                    for number in 0..3usize.pow(len as u32) {
                        let mut prefix = vec![b'a'; len];
                        let mut value = number;
                        for byte in prefix.iter_mut().rev() {
                            *byte += (value % 3) as u8;
                            value /= 3;
                        }
                        let original_cfg_viable = viable.iter().any(|p| p.as_bytes() == prefix);
                        let mut state = c.start();
                        let committed = state.commit_bytes(&prefix).is_ok();
                        assert_eq!(committed, original_cfg_viable,
                            "seed {seed} mode {mode:?}: prefix viability for {prefix:?}");
                        if committed {
                            // The byte-fed and token-fed states must expose
                            // the same future language at EVERY viable
                            // source-CFG prefix, not only complete witnesses.
                            let mut separately_fed = c.start();
                            for &byte in &prefix {
                                separately_fed.commit_token(u32::from(byte - b'a'))
                                    .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?} token-feed prefix {prefix:?}: {e:?}"));
                            }
                            assert_eq!(state.is_accepting(), separately_fed.is_accepting(),
                                "seed {seed} mode {mode:?} byte/token prefix acceptance {prefix:?}");
                            let mask = state.mask();
                            assert_eq!(mask, separately_fed.mask(),
                                "seed {seed} mode {mode:?} byte/token prefix masks {prefix:?}");
                            for (id, token) in token_bytes.into_iter().enumerate() {
                                let mut extended = prefix.clone();
                                extended.extend_from_slice(token);
                                let expected_allowed = viable.iter().any(|p| p.as_bytes() == extended);
                                let actual_allowed = mask.get(id / 32)
                                    .is_some_and(|w| w & (1u32 << (id % 32)) != 0);
                                assert_eq!(actual_allowed, expected_allowed,
                                    "seed {seed} mode {mode:?} mask token {id} after {prefix:?}");
                                if actual_allowed {
                                    let mut by_token = state.clone();
                                    by_token.commit_token(id as u32)
                                        .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?} allowed token {id} after {prefix:?}: {e:?}"));
                                    let mut by_bytes = c.start();
                                    by_bytes.commit_bytes(&extended)
                                        .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?} viable extended prefix {extended:?}: {e:?}"));
                                    assert_eq!(by_token.is_accepting(), by_bytes.is_accepting(),
                                        "seed {seed} mode {mode:?} token-vs-byte continuation acceptance {extended:?}");
                                    assert_eq!(by_token.mask(), by_bytes.mask(),
                                        "seed {seed} mode {mode:?} token-vs-byte continuation mask {extended:?}");
                                }
                            }
                        }
                    }
                }
                for len in 0..=7usize {
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

            // With an explicitly configured EOS token that is absent from
            // the vocabulary, stop admission must agree with the ORIGINAL
            // source grammar's acceptance (not merely with the fast parser).
            let mut eos_module = Module::new(&grammar);
            eos_module.set_mask_strategy(mode).unwrap();
            let with_eos = eos_module
                .compile_with(&vocab, CompileOptions::default().end_tokens([99]))
                .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?}: EOS compilation: {e:?}"));
            let reloaded_eos = glrmask::Constraint::load(with_eos.save())
                .unwrap_or_else(|e| panic!("seed {seed} mode {mode:?}: EOS reload: {e:?}"));
            for eos_constraint in [&with_eos, &reloaded_eos] {
                // Cover completion decisions on 364 complete byte strings,
                // including nullable acceptance and longer recursive paths.
                for len in 0..=5usize {
                    for number in 0..3usize.pow(len as u32) {
                        let mut word = vec![b'a'; len];
                        let mut value = number;
                        for byte in word.iter_mut().rev() {
                            *byte += (value % 3) as u8;
                            value /= 3;
                        }
                        let expected_accepts = expected.iter().any(|x| x.as_bytes() == word);
                        let mut state = eos_constraint.start();
                        let admitted = state.commit_bytes(&word).is_ok();
                        assert_eq!(admitted && state.is_accepting(), expected_accepts,
                            "seed {seed} mode {mode:?} EOS grammar acceptance for {word:?}");
                        if admitted {
                            let mask = state.mask();
                            let eos_is_allowed = mask.get(99 / 32).is_some_and(|w| w & (1u32 << (99 % 32)) != 0);
                            assert_eq!(eos_is_allowed, expected_accepts,
                                "seed {seed} mode {mode:?} EOS mask for {word:?}");
                            if expected_accepts {
                                state.commit_token(99).expect("allowed EOS must commit");
                                assert!(state.is_accepting());
                                assert!(state.mask().iter().all(|&w| w == 0),
                                    "seed {seed} mode {mode:?} EOS did not terminate generation");
                            }
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
