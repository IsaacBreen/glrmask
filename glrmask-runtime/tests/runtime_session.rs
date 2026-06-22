use glrmask::{Constraint, Vocab};
use glrmask_runtime::{RuntimeArtifact, RuntimeConstraint};

fn tiny_constraint() -> Constraint {
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec())],
        None,
    );
    Constraint::from_lark("start: \"a\"", &vocab).unwrap()
}

fn mask(session: &glrmask_runtime::Session, words: usize) -> Vec<u32> {
    let mut result = vec![0; words];
    session.fill_mask(&mut result);
    result
}

#[test]
fn loaded_runtime_constraint_starts_independent_resettable_sessions() {
    let compiled = tiny_constraint();
    let artifact = RuntimeArtifact::from_runtime_payload_v1(compiled.save_runtime_payload_v1());
    let runtime = RuntimeConstraint::from_artifact(artifact).unwrap();

    let mut first = runtime.start();
    let second = runtime.start();
    let initial = mask(&first, runtime.mask_len());
    assert_eq!(initial, mask(&second, runtime.mask_len()));
    assert_eq!(initial[0] & 0b11, 0b01);

    first.commit_token(0).unwrap();
    assert!(first.is_finished());
    assert_eq!(mask(&second, runtime.mask_len()), initial);

    first.reset();
    assert!(!first.is_finished());
    assert_eq!(mask(&first, runtime.mask_len()), initial);
}
