//! Regression tests for two GLR reduce-loop cycles exposed by the TokenTrie
//! runtime mask: a deterministic `replace_top` cycle (seed 2530) and a
//! nondeterministic reduce wave whose frontiers are semantically equal but
//! structurally different (seed 360).
//!
//! The expected masks come from an exact CFG/prefix-DFA intersection over the
//! source CFG, not from any glrmask parser. Masks are checked at every prefix
//! of length at most four, for byte feeds and token feeds.
#[cfg(test)]
use crate::test_modes::MaskMode;

use crate::{Grammar, Module, Vocab};

const TOKEN_BYTES: [&[u8]; 6] = [b"a", b"b", b"c", b"ab", b"bc", b"ca"];

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
        (3, b"ab".to_vec()),
        (4, b"bc".to_vec()),
        (5, b"ca".to_vec()),
    ])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Symbol {
    Byte(u8),
    Nonterminal(usize),
}

struct Cfg {
    start: usize,
    rules: Vec<(usize, Vec<Symbol>)>,
}

fn rule(lhs: usize, rhs: &[Symbol]) -> (usize, Vec<Symbol>) {
    (lhs, rhs.to_vec())
}

/// Exact CYK-style fixpoint over `(nonterminal, from, to)` for a prefix DFA.
///
/// The DFA has states `0..=word.len()`: state `i` means the first `i` bytes of
/// `word` have been matched. For `i < n` only `word[i]` advances to `i + 1`;
/// every other byte has no transition. With `absorbing_final`, state `n` also
/// self-loops on every byte, so the DFA accepts exactly `word..` (the prefix
/// language). Without it, state `n` has no outgoing transition, so the DFA
/// accepts exactly `word`.
///
/// `derives[nt][p][q]` is true iff `nt` derives a terminal string taking the
/// DFA from `p` to `q`. Rules compose as boolean matrix products and the least
/// fixpoint handles epsilon, left recursion and cycles. `(start, 0, n)` is then
/// exactly acceptance (exact DFA) or prefix viability (absorbing final state).
fn dfa_derives(cfg: &Cfg, word: &[u8], absorbing_final: bool) -> bool {
    // Bound 4 plus a two-byte vocabulary token means at most seven DFA states.
    const MAX_STATES: usize = 8;
    let n = word.len();
    let states = n + 1;
    assert!(states <= MAX_STATES, "prefix DFA exceeds the fixed row width");
    let max_nt = cfg
        .rules
        .iter()
        .map(|(lhs, _)| *lhs + 1)
        .max()
        .unwrap_or(0)
        .max(cfg.start + 1);
    // Each relation row is a bitset of destination DFA states.
    let mut derives = vec![[0u64; MAX_STATES]; max_nt];

    loop {
        let mut changed = false;
        for (lhs, rhs) in &cfg.rules {
            let mut relation = [0u64; MAX_STATES];
            for state in 0..states {
                relation[state] = 1u64 << state;
            }
            for symbol in rhs {
                let step = match symbol {
                    Symbol::Byte(byte) => {
                        let mut matrix = [0u64; MAX_STATES];
                        for state in 0..states {
                            if state == n {
                                if absorbing_final {
                                    matrix[state] |= 1u64 << n;
                                }
                            } else if word[state] == *byte {
                                matrix[state] |= 1u64 << (state + 1);
                            }
                        }
                        matrix
                    }
                    Symbol::Nonterminal(nt) => derives[*nt],
                };
                let mut next = [0u64; MAX_STATES];
                for from in 0..states {
                    let mut row = 0u64;
                    let bits = relation[from];
                    for mid in 0..states {
                        if bits & (1u64 << mid) != 0 {
                            row |= step[mid];
                        }
                    }
                    next[from] = row;
                }
                relation = next;
            }
            let target = &mut derives[*lhs];
            for from in 0..states {
                let added = relation[from] & !target[from];
                if added != 0 {
                    target[from] |= added;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    derives[cfg.start][0] & (1u64 << n) != 0
}

fn accepts(cfg: &Cfg, word: &[u8]) -> bool {
    dfa_derives(cfg, word, false)
}

/// A prefix is viable when some accepted word (of any length) extends it.
///
/// This is exact, not bounded: a token is legitimately allowed even when every
/// witness is longer than the four-byte regression bound.
fn viable_prefix(cfg: &Cfg, word: &[u8]) -> bool {
    dfa_derives(cfg, word, true)
}

fn allows(mask: &[u32], token: usize) -> bool {
    mask.get(token / 32)
        .is_some_and(|word| word & (1 << (token % 32)) != 0)
}

/// Expected mask bit: the token is allowed iff the extended prefix is viable.
fn oracle_allows(
    cfg: &Cfg,
    prefix: &[u8],
    token: usize,
    viable_cache: &mut std::collections::HashMap<Vec<u8>, bool>,
) -> bool {
    let mut extended = prefix.to_vec();
    extended.extend_from_slice(TOKEN_BYTES[token]);
    viable_cached(cfg, &extended, viable_cache)
}

fn viable_cached(
    cfg: &Cfg,
    word: &[u8],
    cache: &mut std::collections::HashMap<Vec<u8>, bool>,
) -> bool {
    if let Some(&viable) = cache.get(word) {
        return viable;
    }
    let viable = viable_prefix(cfg, word);
    cache.insert(word.to_vec(), viable);
    viable
}

fn all_words(max_len: usize) -> Vec<Vec<u8>> {
    let mut words = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for prefix in frontier {
            for byte in *b"abc" {
                let mut word = prefix.clone();
                word.push(byte);
                next.push(word);
            }
        }
        words.extend(next.iter().cloned());
        frontier = next;
    }
    words
}

fn tokenize(word: &[u8]) -> Vec<u32> {
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < word.len() {
        let pair = word.get(at..at + 2);
        let token = match pair {
            Some(b"ab") => Some(3),
            Some(b"bc") => Some(4),
            Some(b"ca") => Some(5),
            _ => None,
        };
        if let Some(token) = token {
            tokens.push(token);
            at += 2;
        } else {
            tokens.push(u32::from(word[at] - b'a'));
            at += 1;
        }
    }
    tokens
}

fn check_masks_and_acceptance(cfg: &Cfg, lark: &str, try_auxiliary_backends: bool) {
    const BOUND: usize = 4;
    let vocab = vocab();
    let words = all_words(BOUND);

    // The oracle is independent of the mask backend, so compute each prefix
    // once. `words` contains every string of length at most BOUND, hence every
    // tested prefix.
    let mut expected_masks: std::collections::HashMap<Vec<u8>, [bool; 6]> =
        std::collections::HashMap::new();
    let mut expected_accept = std::collections::HashMap::new();
    let mut expected_viable = std::collections::HashMap::new();
    let mut viable_cache = std::collections::HashMap::new();
    for prefix in &words {
        let bits: [bool; 6] =
            std::array::from_fn(|token| oracle_allows(cfg, prefix, token, &mut viable_cache));
        expected_masks.insert(prefix.clone(), bits);
        expected_accept.insert(prefix.clone(), accepts(cfg, prefix));
        expected_viable.insert(
            prefix.clone(),
            viable_cached(cfg, prefix, &mut viable_cache),
        );
    }

    let mut compiled_modes = Vec::new();
    let mut modes = vec![MaskMode::A];
    if try_auxiliary_backends {
        modes.push(MaskMode::C);
        modes.push(MaskMode::ParserDwa);
    }
    for mode in modes {
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let module = Module::new(&Grammar::from_lark(lark).unwrap());
            let constraint = module
                .compile_with(&vocab, mode.compile_options())
                .ok()?;
            // Some backends decline lazily during materialization rather than
            // returning a compile error. Those modes are simply not exercised.
            let _ = constraint.start();
            Some(constraint)
        }));
        let Some(constraint) = compiled.ok().flatten() else {
            continue;
        };
        compiled_modes.push(mode);

        for prefix in &words {
            if !expected_viable[prefix] {
                assert!(
                    !constraint.start().commit_bytes(prefix),
                    "{mode:?} accepted non-viable prefix {prefix:?}"
                );
                continue;
            }

            // One fresh state per distinct byte prefix is enough: the state and
            // its mask depend only on the committed bytes, not on how the test
            // enumerated them.
            let mut byte_state = constraint.start();
            assert!(byte_state.commit_bytes(prefix));
            let byte_mask = byte_state.mask().to_vec();
            let expected = &expected_masks[prefix];
            for token in 0..TOKEN_BYTES.len() {
                assert_eq!(
                    allows(&byte_mask, token),
                    expected[token],
                    "{mode:?} mask token {token} after {prefix:?}"
                );
            }
            assert_eq!(
                byte_state.is_accepting(),
                expected_accept[prefix],
                "{mode:?} byte-fed acceptance for {prefix:?}"
            );

            // Feed the same prefix through the vocabulary's multi-byte tokens
            // (greedy longest match) and require the identical mask/language.
            let mut token_state = constraint.start();
            for token in tokenize(prefix) {
                assert!(token_state.commit_token(token));
            }
            assert_eq!(
                token_state.mask(),
                byte_mask.as_slice(),
                "{mode:?} byte/token mask disagreement after {prefix:?}"
            );
            assert_eq!(
                token_state.is_accepting(),
                expected_accept[prefix],
                "{mode:?} token-fed acceptance for {prefix:?}"
            );
        }
    }
    assert_eq!(
        compiled_modes.first().copied(),
        Some(MaskMode::A),
        "TokenTrie must compile every regression grammar"
    );
}

// Nonterminals: 0 = start/n3 or n2 as named below, then n0, n1, n2/n3.
//
// seed 2530 minimised:
//   start: n3
//   n0: "c" | n3 |
//   n1: "a" |
//   n2: | n0 n1 n2
//   n3: "b" n2 n0 | n2 "b"
fn seed_2530_minimised() -> Cfg {
    use Symbol::{Byte, Nonterminal as N};
    Cfg {
        start: 0,
        rules: vec![
            rule(0, &[N(4)]),
            rule(1, &[Byte(b'c')]),
            rule(1, &[N(4)]),
            rule(1, &[]),
            rule(2, &[Byte(b'a')]),
            rule(2, &[]),
            rule(3, &[]),
            rule(3, &[N(1), N(2), N(3)]),
            rule(4, &[Byte(b'b'), N(3), N(1)]),
            rule(4, &[N(3), Byte(b'b')]),
        ],
    }
}

const SEED_2530_MINIMISED: &str = "start: n3\nn0: \"c\" | n3 |\nn1: \"a\" |\nn2: | n0 n1 n2\nn3: \"b\" n2 n0 | n2 \"b\"\n";

// seed 2530 full:
//   start: n3
//   n0: "c" | n3 |  |
//   n1: "a" |  | "b" "b" "c" | "c" "c"
//   n2: "a" |  | n0 n1 n2
//   n3: "c" | "b" n2 n0 | n0 "a" n0 | n2 "b" n0
fn seed_2530_full() -> Cfg {
    use Symbol::{Byte, Nonterminal as N};
    Cfg {
        start: 0,
        rules: vec![
            rule(0, &[N(4)]),
            rule(1, &[Byte(b'c')]),
            rule(1, &[N(4)]),
            rule(1, &[]),
            rule(2, &[Byte(b'a')]),
            rule(2, &[]),
            rule(2, &[Byte(b'b'), Byte(b'b'), Byte(b'c')]),
            rule(2, &[Byte(b'c'), Byte(b'c')]),
            rule(3, &[Byte(b'a')]),
            rule(3, &[]),
            rule(3, &[N(1), N(2), N(3)]),
            rule(4, &[Byte(b'c')]),
            rule(4, &[Byte(b'b'), N(3), N(1)]),
            rule(4, &[N(1), Byte(b'a'), N(1)]),
            rule(4, &[N(3), Byte(b'b'), N(1)]),
        ],
    }
}

const SEED_2530_FULL: &str = "start: n3\nn0: \"c\" | n3 |  |\nn1: \"a\" |  | \"b\" \"b\" \"c\" | \"c\" \"c\"\nn2: \"a\" |  | n0 n1 n2\nn3: \"c\" | \"b\" n2 n0 | n0 \"a\" n0 | n2 \"b\" n0\n";

// seed 360 minimised (hung on mask() after committing tokens b, c, c):
//   start: n2
//   n0: "b" |  | n1 "c" n2
//   n1: n1 n0 n0 | n1 n1 |
//   n2: n2 n0 "a" | n0 n1
fn seed_360_minimised() -> Cfg {
    use Symbol::{Byte, Nonterminal as N};
    Cfg {
        start: 0,
        rules: vec![
            rule(0, &[N(3)]),
            rule(1, &[Byte(b'b')]),
            rule(1, &[]),
            rule(1, &[N(2), Byte(b'c'), N(3)]),
            rule(2, &[N(2), N(1), N(1)]),
            rule(2, &[N(2), N(2)]),
            rule(2, &[]),
            rule(3, &[N(3), N(1), Byte(b'a')]),
            rule(3, &[N(1), N(2)]),
        ],
    }
}

const SEED_360_MINIMISED: &str = "start: n2\nn0: \"b\" |  | n1 \"c\" n2\nn1: n1 n0 n0 | n1 n1 |\nn2: n2 n0 \"a\" | n0 n1\n";

// seed 360 full:
//   start: n2
//   n0: "b" |  | n1 "c" n2 |
//   n1: "a" | n1 n0 n0 | n1 n1 |
//   n2: "b" |  | n2 n0 "a" | n0 n1
fn seed_360_full() -> Cfg {
    use Symbol::{Byte, Nonterminal as N};
    Cfg {
        start: 0,
        rules: vec![
            rule(0, &[N(3)]),
            rule(1, &[Byte(b'b')]),
            rule(1, &[]),
            rule(1, &[N(2), Byte(b'c'), N(3)]),
            rule(2, &[Byte(b'a')]),
            rule(2, &[N(2), N(1), N(1)]),
            rule(2, &[N(2), N(2)]),
            rule(2, &[]),
            rule(3, &[Byte(b'b')]),
            rule(3, &[]),
            rule(3, &[N(3), N(1), Byte(b'a')]),
            rule(3, &[N(1), N(2)]),
        ],
    }
}

const SEED_360_FULL: &str = "start: n2\nn0: \"b\" |  | n1 \"c\" n2 |\nn1: \"a\" | n1 n0 n0 | n1 n1 |\nn2: \"b\" |  | n2 n0 \"a\" | n0 n1\n";

#[test]
fn seed_2530_minimised_masks_match_bounded_oracle() {
    check_masks_and_acceptance(&seed_2530_minimised(), SEED_2530_MINIMISED, true);
}

#[test]
fn seed_2530_full_masks_match_bounded_oracle() {
    // The full grammars also decline merged TokenTrie/ParserDwa through the same
    // right-recursive-cycle precondition, but their terminal characterization
    // attempt costs seconds; the minimised tests above exercise that skip path.
    check_masks_and_acceptance(&seed_2530_full(), SEED_2530_FULL, false);
}

#[test]
fn seed_360_minimised_masks_match_bounded_oracle() {
    check_masks_and_acceptance(&seed_360_minimised(), SEED_360_MINIMISED, false);
}

#[test]
fn seed_360_full_masks_match_bounded_oracle() {
    check_masks_and_acceptance(&seed_360_full(), SEED_360_FULL, false);
}
