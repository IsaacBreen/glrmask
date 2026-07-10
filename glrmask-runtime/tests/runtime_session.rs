use glrmask::{Constraint, Vocab};
use glrmask_runtime::{RuntimeArtifact, RuntimeConstraint};

fn tiny_constraint() -> Constraint {
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec())],
        None,
    );
    Constraint::from_glrm_grammar(
        r#"
            start start;
            lexer group a ::= A;
            lexer group b ::= B;
            t A ::= "a";
            t B ::= "b";
            nt start ::= A | B;
        "#,
        &vocab,
    )
    .unwrap()
}

fn legacy_tiny_constraint() -> Constraint {
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec())],
        None,
    );
    Constraint::from_glrm_grammar(
        r#"
            start start;
            lexer group all ::= A, B;
            t A ::= "a";
            t B ::= "b";
            nt start ::= A | B;
        "#,
        &vocab,
    )
    .unwrap()
}

fn mask(session: &glrmask_runtime::Session, words: usize) -> Vec<u32> {
    let mut result = vec![0; words];
    session.fill_mask(&mut result);
    result
}

#[test]
fn loaded_runtime_constraint_starts_independent_resettable_sessions() {
    let compiled = tiny_constraint();
    let artifact = RuntimeArtifact::from_runtime_payload_v2(compiled.save_runtime_payload_v2());
    let runtime = RuntimeConstraint::from_artifact(artifact).unwrap();

    let mut first = runtime.start();
    let second = runtime.start();
    let initial = mask(&first, runtime.mask_len());
    assert_eq!(initial, mask(&second, runtime.mask_len()));
    assert_eq!(initial[0] & 0b11, 0b11);

    first.commit_token(0).unwrap();
    assert!(first.is_finished());
    assert_eq!(mask(&second, runtime.mask_len()), initial);

    first.reset();
    assert!(!first.is_finished());
    assert_eq!(mask(&first, runtime.mask_len()), initial);
}

#[test]
fn current_runtime_loads_legacy_v1_payload() {
    let compiled = legacy_tiny_constraint();
    let artifact = RuntimeArtifact::from_runtime_payload_v1(compiled.save_runtime_payload_v1());
    let runtime = RuntimeConstraint::from_artifact(artifact).unwrap();
    let mut session = runtime.start();
    assert_eq!(mask(&session, runtime.mask_len())[0] & 0b11, 0b11);
    session.commit_token(0).unwrap();
    assert!(session.is_finished());
}

#[test]
fn legacy_v1_envelope_rejects_epsilon_payload() {
    let compiled = tiny_constraint();
    let artifact = RuntimeArtifact::from_runtime_payload_v1(compiled.save_runtime_payload_v2());
    let error = match RuntimeConstraint::from_artifact(artifact) {
        Ok(_) => panic!("legacy v1 envelope accepted an epsilon payload"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("epsilon lexer transitions"));
}
