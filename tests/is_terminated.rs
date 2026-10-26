use glrmask::test_modes::MaskMode;
use glrmask::{
    Constraint, ConstraintState, DynamicConstraint, Expr, Grammar, 
    Module, TokenSet, Vocab};

const STRATEGIES: [MaskMode; 2] = [MaskMode::ParserDwa, MaskMode::A];

fn compile(grammar: &Grammar, vocab: &Vocab, strategy: MaskMode) -> Constraint {
    grammar
        .compile_with(vocab, strategy.compile_options())
        .unwrap()
}

fn with_end_token(body: &Grammar, id: u32) -> Grammar {
    Grammar::from_expr(Expr::sequence([
        Expr::from(body),
        Expr::from(TokenSet::new([id])),
    ]))
    .expect("end-token root expression")
}

fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32)
        .is_some_and(|word| word & (1 << (id % 32)) != 0)
}

/// Independent oracle: can any input still commit? A clone per probe keeps the
/// inspected state untouched.
fn oracle_terminated(state: &ConstraintState<'_>, vocab: &Vocab, special: &[u32]) -> bool {
    for byte in 0..=u8::MAX {
        let mut probe = state.clone();
        if probe.commit_bytes(&[byte]) {
            return false;
        }
    }
    for (id, _) in vocab.iter() {
        let mut probe = state.clone();
        if probe.commit_token(id) {
            return false;
        }
    }
    for &id in special {
        let mut probe = state.clone();
        if probe.commit_token(id) {
            return false;
        }
    }
    true
}

fn assert_oracle(
    constraint: &Constraint,
    vocab: &Vocab,
    special: &[u32],
    accepted: &[u8],
    label: &str,
) {
    let mut state = constraint.start();
    assert_eq!(
        state.is_terminated(),
        oracle_terminated(&state, vocab, special),
        "{label}: prefix 0"
    );
    for (index, &byte) in accepted.iter().enumerate() {
        assert!(state.commit_bytes(&[byte]), "{label}: commit byte {index}");
        assert_eq!(
            state.is_terminated(),
            oracle_terminated(&state, vocab, special),
            "{label}: prefix {} ({})",
            index + 1,
            String::from_utf8_lossy(&accepted[..=index])
        );
    }
}

#[test]
fn complete_match_terminates_and_a_rejected_commit_stays_terminated() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = compile(&Grammar::from_ebnf(r#"start ::= "a""#).unwrap(), &vocab, strategy);
        let mut state = constraint.start();
        assert!(!state.is_accepting());
        assert!(!state.is_terminated());

        assert!(state.commit_bytes(b"a"));
        assert!(state.is_accepting());
        assert!(state.is_terminated());
        assert!(state.mask().iter().all(|word| *word == 0));

        assert!(!state.commit_bytes(b"a"));
        assert!(state.is_terminated());
        assert!(!state.is_accepting());
    }
}

#[test]
fn ignore_lexeme_keeps_an_accepting_state_live() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b" ".to_vec())]);
        let grammar = Grammar::from_lark("start: \"a\"\n%ignore /[ ]+/").unwrap();
        let constraint = compile(&grammar, &vocab, strategy);

        let mut state = constraint.start();
        assert!(!state.is_terminated());
        assert!(state.commit_token(0));
        assert!(state.is_accepting());
        assert!(!state.is_terminated());
        assert!(state.commit_bytes(b" "));
        assert!(!state.is_terminated());
    }
}

#[test]
fn partial_lexeme_is_not_terminated() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())]);
        let constraint = compile(&Grammar::from_ebnf(r#"start ::= "ab""#).unwrap(), &vocab, strategy);
        let mut state = constraint.start();
        assert!(state.commit_bytes(b"a"));
        assert!(!state.is_accepting());
        assert!(!state.is_terminated());
        assert!(state.commit_bytes(b"b"));
        assert!(state.is_accepting());
        assert!(state.is_terminated());
    }
}

#[test]
fn repetition_keeps_an_accepting_state_live() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = compile(&Grammar::from_ebnf(r#"start ::= "a"+"#).unwrap(), &vocab, strategy);
        let mut state = constraint.start();
        assert!(state.commit_token(0));
        assert!(state.is_accepting());
        assert!(!state.is_terminated());
        assert!(state.commit_token(0));
        assert!(state.is_accepting());
        assert!(!state.is_terminated());
    }
}

#[test]
fn end_token_root_terminates_only_after_the_end_token() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]).with_special_tokens([99]);
        let grammar = with_end_token(&Grammar::from_ebnf(r#"start ::= "a""#).unwrap(), 99);
        let constraint = compile(&grammar, &vocab, strategy);

        let mut state = constraint.start();
        assert!(!state.is_terminated());
        assert!(state.commit_token(0));
        assert!(!state.is_accepting());
        assert!(!state.is_terminated());
        assert!(allowed(&state.mask(), 99));

        assert!(state.commit_token(99));
        assert!(state.is_accepting());
        assert!(state.is_terminated());
        assert!(state.mask().iter().all(|word| *word == 0));
    }
}

#[test]
fn special_token_on_the_normal_path_keeps_the_state_live() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]).with_special_tokens([99]);
        let grammar = Grammar::from_glrm(
            "glrm 1; start root; extern token eos; nt root ::= \"a\" eos;",
        ).unwrap();
        let mut module = Module::new(&grammar);
        module.bind("eos", TokenSet::new([99])).unwrap();
        let constraint = module
            .compile_with(&vocab, strategy.compile_options())
            .unwrap();

        let mut state = constraint.start();
        assert!(!state.is_terminated());
        assert!(state.commit_token(0));
        assert!(!state.is_accepting());
        // The only remaining input is the zero-width special token; probe on a
        // clone because a rejected commit kills the state.
        assert!(!state.clone().commit_bytes(b"a"));
        assert!(!state.is_terminated());
        assert!(state.commit_token(99));
        assert!(state.is_accepting());
        assert!(state.is_terminated());
    }
}

#[test]
fn ignored_root_with_a_special_token_stays_live() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b" ".to_vec())])
            .with_special_tokens([99]);
        let grammar = Grammar::from_glrm(
            r#"
                glrm 1;
                start root;
                ignore WS;
                t WS ::= " "+;
                extern token eos;
                nt root ::= "a" eos;
            "#,
        ).unwrap();
        let mut module = Module::new(&grammar);
        module.bind("eos", TokenSet::new([99])).unwrap();
        let constraint = module
            .compile_with(&vocab, strategy.compile_options())
            .unwrap();

        let mut state = constraint.start();
        assert!(state.commit_token(0));
        assert!(!state.is_terminated());
        assert!(state.commit_token(99));
        assert!(state.is_accepting());
        // Trailing ignore is still committable after the grammar completes.
        assert!(!state.is_terminated());
        assert!(state.commit_bytes(b" "));
    }
}

#[test]
fn oracle_agrees_with_commit_for_every_prefix() {
    let vocab_a = Vocab::new(vec![(0, b"a".to_vec())]).with_special_tokens([99]);
    let json_vocab = Vocab::new(vec![
        (0, b"{".to_vec()),
        (1, b"}".to_vec()),
        (2, b",".to_vec()),
        (3, b":".to_vec()),
        (4, b"\"k\"".to_vec()),
        (5, b"0".to_vec()),
        (6, b"1".to_vec()),
        (7, b" ".to_vec()),
    ])
    .with_special_tokens([99]);
    let plain_vocab = Vocab::new(vec![(0, b"a".to_vec())]);

    let json_body = Grammar::from_lark(
        "start: \"{\" pair (\",\" pair)* \"}\"\npair: \"\\\"k\\\"\" \":\" (\"0\"|\"1\")\n%ignore /[ ]+/",
    ).unwrap();
    let json_root = with_end_token(&json_body, 99);
    let ignore_body = Grammar::from_lark("start: \"a\"\n%ignore /[ ]+/").unwrap();

    for strategy in STRATEGIES {
        assert_oracle(
            &compile(&Grammar::from_ebnf(r#"start ::= "a""#).unwrap(), &plain_vocab, strategy),
            &plain_vocab,
            &[],
            b"a",
            "plain a",
        );
        assert_oracle(
            &compile(&Grammar::from_ebnf(r#"start ::= "a"+"#).unwrap(), &plain_vocab, strategy),
            &plain_vocab,
            &[],
            b"aaa",
            "repetition",
        );
        assert_oracle(
            &compile(&ignore_body, &vocab_a, strategy),
            &vocab_a,
            &[],
            b"a",
            "ignore",
        );
        assert_oracle(
            &compile(&json_root, &json_vocab, strategy),
            &json_vocab,
            &[99],
            br#"{"k":0,"k":1}"#,
            "json with end token",
        );
    }
}

#[test]
fn dynamic_runtime_oracle_agrees_with_commit_for_every_prefix() {
    let plain_vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let ignore_vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b" ".to_vec())]);

    let plain = Grammar::from_ebnf(r#"start ::= "a""#).unwrap();
    let repetition = Grammar::from_ebnf(r#"start ::= "a"+"#).unwrap();
    let ignore = Grammar::from_lark("start: \"a\"\n%ignore /[ ]+/").unwrap();
    let cases: [(&str, &Grammar, &Vocab, &[u8]); 3] = [
        ("plain a", &plain, &plain_vocab, b"a"),
        ("repetition", &repetition, &plain_vocab, b"aaa"),
        ("ignore", &ignore, &ignore_vocab, b"a"),
    ];

    for (label, grammar, vocab, accepted) in cases {
        let constraint = DynamicConstraint::compile(grammar.clone(), vocab).unwrap();
        let mut state = constraint.start();
        assert_eq!(
            state.is_terminated(),
            dynamic_oracle(&state, vocab),
            "{label}: prefix 0"
        );
        for (index, &byte) in accepted.iter().enumerate() {
            assert!(state.commit_bytes(&[byte]), "{label}: byte {index}");
            assert_eq!(
                state.is_terminated(),
                dynamic_oracle(&state, vocab),
                "{label}: prefix {}",
                index + 1
            );
        }
    }
}

fn dynamic_oracle(state: &glrmask::DynamicConstraintState<'_>, vocab: &Vocab) -> bool {
    for byte in 0..=u8::MAX {
        let mut probe = state.clone();
        if probe.commit_bytes(&[byte]) {
            return false;
        }
    }
    for (id, _) in vocab.iter() {
        let mut probe = state.clone();
        if probe.commit_token(id) {
            return false;
        }
    }
    true
}

#[test]
fn empty_commit_on_failed_state_returns_false() {
    for strategy in STRATEGIES {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = compile(&Grammar::from_ebnf(r#"start ::= "a""#).unwrap(), &vocab, strategy);

        // Fresh state: both empty commits return true
        let mut fresh = constraint.start();
        assert!(!fresh.is_terminated());
        assert!(!fresh.is_accepting());
        assert!(fresh.commit_bytes(b""));
        assert!(fresh.commit_tokens(&[]));

        // Finished state (terminated and accepting): both empty commits return true
        let mut finished = constraint.start();
        assert!(finished.commit_bytes(b"a"));
        assert!(finished.is_accepting());
        assert!(finished.is_terminated());
        assert!(finished.commit_bytes(b""));
        assert!(finished.commit_tokens(&[]));

        // Failed state: committing a disallowed token
        let mut failed = constraint.start();
        assert!(!failed.commit_token(999));
        assert!(failed.is_terminated());
        assert!(!failed.is_accepting());
        assert!(!failed.commit_bytes(b""));
        assert!(failed.is_terminated());
        assert!(!failed.is_accepting());
        assert!(!failed.commit_tokens(&[]));
        assert!(failed.is_terminated());
        assert!(!failed.is_accepting());
    }
}

