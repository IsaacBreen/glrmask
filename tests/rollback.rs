use glrmask::{Constraint, ConstraintState, Vocab};

fn vocab(entries: &[&str], eos_token_id: Option<u32>) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, text)| (id as u32, text.as_bytes().to_vec()))
            .collect(),
        eos_token_id,
    )
}

fn constraint(entries: &[&str], grammar: &str) -> Constraint {
    Constraint::from_ebnf(grammar, &vocab(entries, None)).unwrap()
}

fn mask(state: &ConstraintState<'_>) -> Vec<u32> {
    state.mask()
}

#[test]
fn rollback_restores_exact_mask_and_completion() {
    let constraint = constraint(&["a", "b", "c"], r#"start ::= "a" ("b" | "c")"#);
    let mut state = constraint.start_with_rollback(4);

    let initial_mask = mask(&state);
    state.commit_token(0).unwrap();
    let after_a_mask = mask(&state);
    assert!(!state.is_complete());

    state.commit_token(1).unwrap();
    assert!(state.is_complete());

    state.rollback(1).unwrap();
    assert_eq!(mask(&state), after_a_mask);
    assert!(!state.is_complete());

    state.rollback(1).unwrap();
    assert_eq!(mask(&state), initial_mask);
    assert!(!state.is_complete());
}

#[test]
fn repeated_speculative_cycles_do_not_accumulate_state() {
    let constraint = constraint(&["a", "b", "c"], r#"start ::= "a" ("b" | "c")"#);
    let mut state = constraint.start_with_rollback(2);
    let initial_mask = mask(&state);

    for choice in [1, 2, 1, 2, 1] {
        state.commit_token(0).unwrap();
        state.commit_token(choice).unwrap();
        assert!(state.is_complete());
        state.rollback(2).unwrap();
        assert_eq!(mask(&state), initial_mask);
        assert!(!state.is_complete());
    }
}

#[test]
fn validate_tokens_is_non_destructive_and_stops_at_invalid_suffix() {
    let constraint = constraint(&["a", "b", "x"], r#"start ::= "a" "b""#);
    let state = constraint.start_with_rollback(4);
    let initial_mask = mask(&state);

    assert_eq!(state.validate_tokens(&[0, 1]), vec![0, 1]);
    assert_eq!(mask(&state), initial_mask);

    assert_eq!(state.validate_tokens(&[0, 2, 1]), vec![0]);
    assert_eq!(mask(&state), initial_mask);
    assert!(!state.is_complete());
}

#[test]
fn invalid_commit_can_be_rolled_back_without_contamination() {
    let constraint = constraint(&["a", "b", "x"], r#"start ::= "a" "b""#);
    let mut state = constraint.start_with_rollback(2);

    state.commit_token(0).unwrap();
    let after_a_mask = mask(&state);

    // Existing commit semantics allow a grammatical rejection to drive the
    // state into a fail state. Rollback must still restore the exact prior
    // state because the pre-commit snapshot is retained.
    let _ = state.commit_token(2);
    assert!(state.is_failed());

    state.rollback(1).unwrap();
    assert!(!state.is_failed());
    assert_eq!(mask(&state), after_a_mask);
}

#[test]
fn rollback_history_is_bounded_and_fails_clearly() {
    let constraint = constraint(
        &["a"],
        r#"start ::= "a" "a" "a" "a""#,
    );
    let mut state = constraint.start_with_rollback(2);

    state.commit_token(0).unwrap();
    let after_one = mask(&state);
    state.commit_token(0).unwrap();
    state.commit_token(0).unwrap();

    let err = state.rollback(3).unwrap_err();
    assert!(err.contains("only 2 retained"), "{err}");

    state.rollback(2).unwrap();
    assert_eq!(mask(&state), after_one);
}

#[test]
fn unknown_token_error_does_not_consume_rollback_history() {
    let constraint = constraint(&["a", "b"], r#"start ::= "a" "b""#);
    let mut state = constraint.start_with_rollback(2);
    let initial_mask = mask(&state);

    state.commit_token(0).unwrap();
    assert!(state.commit_token(999).is_err());

    // The failed out-of-vocabulary attempt did not mutate semantic state and
    // must not become an artificial rollback event. The one retained event is
    // still the successful commit of `a`.
    state.rollback(1).unwrap();
    assert_eq!(mask(&state), initial_mask);
}

#[test]
fn rollback_crosses_completion_and_eos_admissibility_boundary() {
    let vocab = vocab(&["a", "<eos>"], Some(1));
    let constraint = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
    let mut state = constraint.start_with_rollback(1);
    let initial_mask = mask(&state);

    state.commit_token(0).unwrap();
    assert!(state.is_complete());
    let complete_mask = mask(&state);
    assert_ne!(complete_mask, initial_mask);
    assert_ne!(complete_mask[0] & (1 << 1), 0, "EOS should be admissible");

    state.rollback(1).unwrap();
    assert!(!state.is_complete());
    assert_eq!(mask(&state), initial_mask);
}

#[test]
fn independent_states_share_constraint_without_state_contamination() {
    let constraint = constraint(&["a", "b", "c"], r#"start ::= "a" ("b" | "c")"#);
    let mut left = constraint.start_with_rollback(2);
    let mut right = constraint.start_with_rollback(2);

    left.commit_token(0).unwrap();
    left.commit_token(1).unwrap();
    assert!(left.is_complete());

    assert!(!right.is_complete());
    right.commit_token(0).unwrap();
    let right_after_a = mask(&right);

    left.rollback(2).unwrap();
    assert!(!left.is_complete());
    assert_eq!(mask(&right), right_after_a);
}
