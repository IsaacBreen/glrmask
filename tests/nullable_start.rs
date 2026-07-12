use glrmask::__private::ConstraintExt as _;
use glrmask::{Constraint, DynamicConstraint, Vocab};

const EOS: u32 = 7;

fn vocab() -> Vocab {
    Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b" ".to_vec()),
            (4, b"  ".to_vec()),
            (5, b"\n".to_vec()),
            (6, b"\n\n".to_vec()),
            (EOS, b"<eos>".to_vec()),
        ],
        Some(EOS),
    )
}

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    mask.get(token_id as usize / 32)
        .is_some_and(|word| word & (1u32 << (token_id % 32)) != 0)
}

fn assert_complete_with_eos(constraint: &Constraint, bytes: &[u8]) {
    let mut state = constraint.start();
    state.commit_bytes(bytes).unwrap();
    assert!(state.is_complete(), "expected {bytes:?} to be complete");
    assert!(token_allowed(&state.mask(), EOS), "EOS missing for {bytes:?}");
}

#[test]
fn pure_epsilon_start_is_complete_without_a_commit() {
    let grammar = r#"
        start empty;
        nt empty ::= eps;
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab()).unwrap();
    let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab()).unwrap();

    assert_complete_with_eos(&constraint, b"");
    assert!(constraint.start().force().is_empty());
    let dynamic_state = dynamic.start();
    assert!(dynamic_state.is_complete());
    assert!(token_allowed(&dynamic_state.mask(), EOS));
    assert!(dynamic_state.force().is_empty());
}

#[test]
fn nullable_start_does_not_make_a_partial_nonempty_branch_complete() {
    let grammar = r#"
        start value;
        nt value ::= eps | "a" "b";
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab()).unwrap();

    assert_complete_with_eos(&constraint, b"");
    assert_complete_with_eos(&constraint, b"ab");

    let mut partial = constraint.start();
    partial.commit_bytes(b"a").unwrap();
    assert!(!partial.is_complete());
    assert!(!token_allowed(&partial.mask(), EOS));

    let mut rejected = constraint.start();
    assert!(rejected.commit_bytes(b"b").is_err());
}

#[test]
fn ignore_only_input_preserves_nullable_start_completion() {
    let grammar = r#"
        start empty;
        ignore WS;
        t WS ::= " "+;
        nt empty ::= eps;
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab()).unwrap();
    let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab()).unwrap();

    for bytes in [b"".as_slice(), b" ".as_slice(), b"    ".as_slice()] {
        assert_complete_with_eos(&constraint, bytes);

        let mut state = dynamic.start();
        state.commit_bytes(bytes).unwrap();
        assert!(state.is_complete(), "dynamic state incomplete for {bytes:?}");
        assert!(token_allowed(&state.mask(), EOS));
    }
}

#[test]
fn nullable_terminal_root_preserves_empty_acceptance_before_epsilon_elimination() {
    let grammar = r#"
        start value;
        t MAYBE_A ::= "a"*;
        nt value ::= MAYBE_A;
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab()).unwrap();

    assert_complete_with_eos(&constraint, b"");
    assert_complete_with_eos(&constraint, b"a");
}

#[test]
fn nullable_subgrammar_with_local_ignore_accepts_empty_and_ignore_only_input() {
    let grammar = r#"
        start document;

        g inner ::= {
            start empty;
            ignore NL;
            t NL ::= "\n"+;
            nt empty ::= eps;
        };

        nt document ::= inner;
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab()).unwrap();

    assert_complete_with_eos(&constraint, b"");
    assert_complete_with_eos(&constraint, b"\n\n");
}

#[test]
fn nullable_start_roundtrips_all_current_artifact_formats() {
    let grammar = r#"
        start empty;
        ignore WS;
        t WS ::= " "+;
        nt empty ::= eps;
    "#;
    let constraint = Constraint::from_glrm_grammar(grammar, &vocab()).unwrap();

    let loaded = Constraint::load(&constraint.save()).unwrap();
    assert_complete_with_eos(&loaded, b"");
    assert_complete_with_eos(&loaded, b"  ");

    let runtime = Constraint::load_runtime_payload_v4(&constraint.save_runtime_payload_v4()).unwrap();
    assert_complete_with_eos(&runtime, b"");
    assert_complete_with_eos(&runtime, b"  ");

    for save_old in [
        Constraint::save_runtime_payload_v1 as fn(&Constraint) -> Vec<u8>,
        Constraint::save_runtime_payload_v2,
        Constraint::save_runtime_payload_v3,
    ] {
        assert!(
            std::panic::catch_unwind(|| save_old(&constraint)).is_err(),
            "an old runtime payload silently dropped nullable-start metadata",
        );
    }

    let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab()).unwrap();
    let dynamic_loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
    let mut state = dynamic_loaded.start();
    state.commit_bytes(b"  ").unwrap();
    assert!(state.is_complete());
    assert!(token_allowed(&state.mask(), EOS));
}
