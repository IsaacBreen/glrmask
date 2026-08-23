use glrmask::{CompileOptions, DynamicConstraint, Grammar, StaticConstraint, Vocab};

#[test]
fn termination_is_caller_policy_after_acceptance() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (64, Vec::new())]);
    let grammar = Grammar::ebnf(r#"start ::= "a""#);
    let options = CompileOptions::default();
    let static_constraint = StaticConstraint::compile(grammar.clone(), &vocab, &options).unwrap();
    let dynamic_constraint = DynamicConstraint::compile(grammar, &vocab, &options).unwrap();

    let mut static_state = static_constraint.start();
    let mut dynamic_state = dynamic_constraint.start();
    static_state.commit_token(0).unwrap();
    dynamic_state.commit_token(0).unwrap();
    assert!(static_state.is_accepting());
    assert!(dynamic_state.is_accepting());

    // The constraint has no EOS concept. A caller stops here (or applies its
    // own generation policy) based only on acceptance; token 64 is never
    // handed to the constraint as an implicit terminator.
    let caller_should_stop = static_state.is_accepting() && dynamic_state.is_accepting();
    assert!(caller_should_stop);
}
