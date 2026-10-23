//! Regression coverage for the seed-3229 TokenTrie stall caused by
//! identity unit rules (`X -> X`) created during epsilon elimination.

use std::sync::mpsc;
use std::time::Duration;

use glrmask::{CompileOptions, Grammar, MaskStrategy, Module, Vocab};

const ORIGINAL: &str = r#"start: n3
n0: "a" | n1 "c" n1 | "a" n3 | "b" n1 n1
n1: "c" | n2 n1
n2: "c" | "c" n0 |
n3: "b" | n2 "a"
"#;

const MINIMIZED: &str = r#"start: n3
n0: n1 "c" n1
n1: "c" | n2 n1
n2: "c" n0 |
n3: n2 "a"
"#;

const NULLABLE_TWO_CYCLE: &str = r#"start: a
a: n b | "x"
b: n a | "y"
n: "z" |
"#;

const MODES: [MaskStrategy; 4] = [
    MaskStrategy::Auto,
    MaskStrategy::ParserDwa,
    MaskStrategy::TokenTrie,
    MaskStrategy::TokenClassTrie,
];

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec()),
        (3, b"ab".to_vec()), (4, b"bc".to_vec()), (5, b"ca".to_vec()),
    ])
}

fn xyz_vocab() -> Vocab {
    Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec()), (2, b"z".to_vec())])
}

/// Commits `tokens` on a worker thread. Returns Ok(Some(accepting)) when every
/// commit succeeds, Ok(None) when a commit is rejected, and Err on a stall.
fn commit_tokens(
    lark: &'static str,
    mode: MaskStrategy,
    vocab_fn: fn() -> Vocab,
    tokens: &'static [u32],
) -> Result<Option<bool>, String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let module = Module::new(&Grammar::from_lark(lark));
        let constraint = module
            .compile_with(&vocab_fn(), CompileOptions::default().mask_strategy(mode))
            .unwrap();
        let mut state = constraint.start();
        for &token in tokens {
            let _ = state.mask();
            if state.commit_token(token).is_err() {
                tx.send(None).ok();
                return;
            }
        }
        tx.send(Some(state.is_accepting())).ok();
    });
    rx.recv_timeout(Duration::from_secs(20))
        .map_err(|_| format!("{mode:?}: committing {tokens:?} did not finish within 20s"))
}

#[test]
fn seed3229_valid_prefix_commits_token_by_token() {
    for mode in MODES {
        // `cca` is a prefix of `ccaccca`; both must commit, the latter accepts.
        assert_eq!(commit_tokens(ORIGINAL, mode, vocab, &[2, 2, 0]), Ok(Some(false)), "{mode:?} cca");
        assert_eq!(commit_tokens(ORIGINAL, mode, vocab, &[2, 2, 0, 2, 2, 2, 0]), Ok(Some(true)), "{mode:?} ccaccca");
    }
}

#[test]
fn seed3229_minimized_grammar_does_not_stall() {
    for mode in MODES {
        assert!(commit_tokens(MINIMIZED, mode, vocab, &[2, 2, 0]).is_ok(), "{mode:?} stalled");
    }
}

#[test]
fn unit_cycle_through_nullable_prefix_does_not_stall() {
    // a and b reach each other through nullable n, so after epsilon removal
    // a -> b and b -> a form a unit cycle. Language: z* (x | y).
    for mode in MODES {
        for (tokens, expected) in [
            (&[0u32][..], Some(true)),
            (&[1][..], Some(true)),
            (&[2, 0][..], Some(true)),
            (&[2, 2, 1][..], Some(true)),
            (&[2][..], Some(false)),
            (&[0, 2][..], None),
        ] {
            assert_eq!(
                commit_tokens(NULLABLE_TWO_CYCLE, mode, xyz_vocab, tokens),
                Ok(expected),
                "{mode:?} {tokens:?}"
            );
        }
    }
}
