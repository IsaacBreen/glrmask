use glrmask::{Constraint, Vocab};

fn setup() -> (Constraint, Vec<u32>) {
    let vocab = Vocab::new(
        vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"x".to_vec()),
        ],
        None,
    );
    let constraint = Constraint::from_ebnf(r#"start ::= "a" "b""#, &vocab).unwrap();
    let initial_mask = constraint.start().mask();
    (constraint, initial_mask)
}

#[test]
fn validate_tokens_is_non_mutating_and_returns_longest_prefix() {
    let (constraint, initial_mask) = setup();
    let state = constraint.start_with_rollback(2);
    assert_eq!(state.validate_tokens(&[0, 1, 2]), vec![0, 1]);
    assert_eq!(state.mask(), initial_mask);
    assert!(!state.is_complete());
}

#[test]
fn rollback_restores_mask_and_completion() {
    let (constraint, initial_mask) = setup();
    let mut state = constraint.start_with_rollback(2);
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    assert!(state.is_complete());
    state.rollback(2).unwrap();
    assert_eq!(state.mask(), initial_mask);
    assert!(!state.is_complete());
    assert!(!state.is_failed());
}

#[test]
fn invalid_commit_can_be_rolled_back() {
    let (constraint, initial_mask) = setup();
    let mut state = constraint.start_with_rollback(1);
    assert!(state.commit_token(2).is_err());
    assert!(state.is_failed());
    state.rollback(1).unwrap();
    assert_eq!(state.mask(), initial_mask);
    assert!(!state.is_failed());
}

#[test]
fn unknown_token_does_not_consume_history() {
    let (constraint, initial_mask) = setup();
    let mut state = constraint.start_with_rollback(1);
    state.commit_token(0).unwrap();
    assert!(state.commit_token(999).is_err());
    state.rollback(1).unwrap();
    assert_eq!(state.mask(), initial_mask);
}

#[test]
fn rollback_failure_is_atomic() {
    let (constraint, _) = setup();
    let mut state = constraint.start_with_rollback(1);
    state.commit_token(0).unwrap();
    let mask = state.mask();
    assert!(state.rollback(2).is_err());
    assert_eq!(state.mask(), mask);
    state.rollback(1).unwrap();
}

#[test]
fn history_is_bounded() {
    let (constraint, _) = setup();
    let mut state = constraint.start_with_rollback(1);
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    state.rollback(1).unwrap();
    assert!(!state.is_complete());
    assert!(state.rollback(1).is_err());
}

#[test]
fn zero_history_retains_no_rollback_state() {
    let (constraint, _) = setup();
    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    assert!(state.rollback(1).is_err());
}

#[test]
fn repeated_speculative_cycles_restore_exactly() {
    let (constraint, initial_mask) = setup();
    let mut state = constraint.start_with_rollback(2);
    for _ in 0..4 {
        assert_eq!(state.validate_tokens(&[0, 1]), vec![0, 1]);
        state.commit_token(0).unwrap();
        state.commit_token(1).unwrap();
        state.rollback(2).unwrap();
        assert_eq!(state.mask(), initial_mask);
    }
}
