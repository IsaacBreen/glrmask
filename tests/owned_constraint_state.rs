#![recursion_limit = "512"]

use glrmask::{Constraint, ConstraintState, Grammar, Module, TokenSet, Vocab};

#[test]
fn public_types_are_send_sync_clone() {
    fn assert_traits<T: Send + Sync + Clone>() {}
    assert_traits::<Grammar>();
    assert_traits::<Module>();
    assert_traits::<Constraint>();
    assert_traits::<ConstraintState>();
    assert_traits::<Vocab>();
    assert_traits::<TokenSet>();
}

#[test]
fn a_state_outlives_its_constraint_variable() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec())]);
    let mut state = {
        let c = Constraint::compile(Grammar::json_schema(r#"{"enum":["x"]}"#), &vocab)
            .unwrap();
        c.start()
    };
    assert!(state.commit_bytes(b"\"x\""));
    assert!(state.is_accepting());
}
