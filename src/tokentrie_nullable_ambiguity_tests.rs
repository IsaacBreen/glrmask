//! TokenTrie must terminate on infinitely ambiguous nullable grammars.
//!
//! `S -> "a" | S S | ""` has infinitely many derivations for every `a`+ word.
//! Committing one `a` leaves a parser state whose epsilon/GLR bookkeeping is
//! cyclic; the next mask or byte commit must merge equivalent stack
//! configurations instead of regenerating them forever.

use std::sync::mpsc;
use std::time::Duration;

use crate::{CompileOptions, Grammar, MaskStrategy, Module, Vocab};

const NULLABLE_AMBIGUOUS: &str = "start: n0\nn0: \"a\" | n0 n0 |\n";

fn vocab() -> Vocab {
    Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())])
}

fn token_trie() -> crate::runtime::Constraint {
    let module = Module::new(&Grammar::from_lark(NULLABLE_AMBIGUOUS).unwrap());
    module
        .compile_with(&vocab(), CompileOptions::default().mask_strategy(MaskStrategy::TokenTrie))
        .unwrap()
}

fn mode_constraint(mode: MaskStrategy) -> crate::runtime::Constraint {
    let module = Module::new(&Grammar::from_lark(NULLABLE_AMBIGUOUS).unwrap());
    module
        .compile_with(&vocab(), CompileOptions::default().mask_strategy(mode))
        .unwrap()
}

/// Run `body` on a worker thread; panic if it does not finish in 10 s.
fn within_deadline<T: Send + 'static>(what: &str, body: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        tx.send(body()).ok();
    });
    rx.recv_timeout(Duration::from_secs(10))
        .unwrap_or_else(|_| panic!("{what} did not finish within 10s"))
}

#[test]
fn tokentrie_nullable_ambiguity_mask_and_byte_commit_terminate() {
    let outcome = within_deadline("TokenTrie mask/commit after one 'a'", || {
        let constraint = token_trie();
        let mut state = constraint.start();
        assert!(state.commit_token(0), "commit first 'a'");
        let mask = state.mask();
        assert!(state.commit_bytes(b"a"), "commit second 'a'");
        (mask, state.is_accepting())
    });
    assert!(
        outcome.0[0] & 1 != 0,
        "mask after one 'a' must still allow 'a'"
    );
    assert!(outcome.1, "\"aa\" must be accepted");
}

#[test]
fn tokentrie_masks_match_parser_dwa_for_nullable_ambiguity() {
    let words = within_deadline("cross-mode comparison", || {
        let trie = token_trie();
        let dwa = mode_constraint(MaskStrategy::ParserDwa);
        let mut diffs = Vec::new();
        let mut words = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..5 {
            let mut next = Vec::new();
            for prefix in frontier {
                for byte in [b'a', b'b'] {
                    let mut word = prefix.clone();
                    word.push(byte);
                    next.push(word);
                }
            }
            words.extend(next.iter().cloned());
            frontier = next;
        }
        for word in words {
            let mut left = trie.start();
            let mut right = dwa.start();
            let mut prefix = Vec::new();
            for byte in word {
                let mut a = vec![0u32; left.mask().len()];
                let mut b = vec![0u32; right.mask().len()];
                left.fill_mask(&mut a);
                right.fill_mask(&mut b);
                if a != b {
                    diffs.push((prefix.clone(), a, b));
                    break;
                }
                if !left.commit_bytes(&[byte]) {
                    assert!(!right.commit_bytes(&[byte]), "rejection disagreement {prefix:?}");
                    break;
                }
                assert!(right.commit_bytes(&[byte]), "ParserDwa must accept what TokenTrie accepts");
                prefix.push(byte);
            }
            assert_eq!(left.is_accepting(), right.is_accepting(), "acceptance disagreement {prefix:?}");
            let mut a = vec![0u32; left.mask().len()];
            let mut b = vec![0u32; right.mask().len()];
            left.fill_mask(&mut a);
            right.fill_mask(&mut b);
            assert_eq!(a, b, "final mask disagreement {prefix:?}");
        }
        diffs
    });
    assert!(words.is_empty(), "mask differences at prefixes {words:?}");
}
