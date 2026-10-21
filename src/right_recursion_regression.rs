//! Regression coverage for right-recursion elimination on a small grammar that
//! used to grow the rule set without bound.

use crate::{Grammar, MaskStrategy, Module, Vocab};
use std::collections::BTreeSet;

/// Minimal grammar that previously made `eliminate_right_recursion` expand
/// exponentially (the surviving subset of seed 349).
///
/// ```text
/// 0 -> "b" 0 0
/// 1 -> "c" | 2
/// 2 -> ε   | 1 2
/// ```
const GRAMMAR: &str = "start: n0\nn0: \"b\" n0 n0\nn1: \"c\" | n2\nn2:  | n1 n2\n";

const BOUND: usize = 3;
const ALPHABET: usize = 3;

fn language() -> BTreeSet<Vec<u32>> {
    // Flat form of GRAMMAR. Negative symbols are terminals: -1 = 'a', etc.
    const RULES: &[(u32, &[i32])] = &[
        (0, &[-2, 0, 0]),
        (1, &[-3]),
        (1, &[2]),
        (2, &[]),
        (2, &[1, 2]),
    ];
    let mut languages = vec![BTreeSet::<Vec<u32>>::new(); 3];
    loop {
        let mut changed = false;
        for &(lhs, rhs) in RULES {
            let mut words = BTreeSet::from([Vec::new()]);
            for &symbol in rhs {
                let suffixes = if symbol < 0 {
                    BTreeSet::from([vec![(-symbol - 1) as u32]])
                } else {
                    languages[symbol as usize].clone()
                };
                let mut joined = BTreeSet::new();
                for prefix in &words {
                    for suffix in &suffixes {
                        if prefix.len() + suffix.len() <= BOUND {
                            let mut word = prefix.clone();
                            word.extend_from_slice(suffix);
                            joined.insert(word);
                        }
                    }
                }
                words = joined;
            }
            let before = languages[lhs as usize].len();
            languages[lhs as usize].extend(words);
            changed |= languages[lhs as usize].len() != before;
        }
        if !changed {
            return languages.swap_remove(0);
        }
    }
}

fn allowed(mask: &[u32], token: u32) -> bool {
    mask.get(token as usize / 32)
        .is_some_and(|word| word & (1 << (token % 32)) != 0)
}

fn words() -> Vec<Vec<u32>> {
    let mut all = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    for _ in 0..BOUND {
        let mut next = Vec::new();
        for word in frontier {
            for token in 0..ALPHABET as u32 {
                let mut extended = word.clone();
                extended.push(token);
                next.push(extended);
            }
        }
        all.extend(next.clone());
        frontier = next;
    }
    all
}

#[test]
fn minimal_right_recursion_grammar_compiles_and_masks_match_the_language() {
    let expected = language();
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"c".to_vec()),
    ]);
    let grammar = Grammar::from_lark(GRAMMAR);
    for strategy in [
        MaskStrategy::TokenTrie,
        MaskStrategy::TokenClassTrie,
        MaskStrategy::ParserDwa,
    ] {
        let mut module = Module::new(&grammar);
        module.set_mask_strategy(strategy).unwrap();
        let constraint = module.compile(&vocab).unwrap();

        for word in words() {
            let mut state = constraint.start();
            let mut accepted_prefix = true;
            for &token in &word {
                if state.commit_token(token).is_err() {
                    accepted_prefix = false;
                    break;
                }
            }
            if !accepted_prefix {
                assert!(
                    !expected.contains(&word),
                    "{strategy:?} rejected valid word {word:?}",
                );
                continue;
            }
            assert_eq!(
                state.is_accepting(),
                expected.contains(&word),
                "{strategy:?} acceptance for {word:?}",
            );
            let mask = state.mask();
            for token in 0..ALPHABET as u32 {
                let mut extended = word.clone();
                extended.push(token);
                if expected.contains(&extended) {
                    assert!(
                        allowed(&mask, token),
                        "{strategy:?} mask after {word:?} hides {token}",
                    );
                }
            }
        }
    }
}
