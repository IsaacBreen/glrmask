use glrmask::{CompileOptions, Constraint, DynamicConstraint, Grammar, Vocab};

fn allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word)
        .is_some_and(|value| value & (1u32 << bit) != 0)
}

fn assert_static(mut state: glrmask::ConstraintState<'_>) {
    assert!(allowed(&state.mask(), 0));
    assert!(!allowed(&state.mask(), 64));
    state.commit_token(0).unwrap();
    assert!(!state.is_accepting());
    assert!(allowed(&state.mask(), 64));
    state.commit_token(64).unwrap();
    assert!(state.is_accepting());
}

fn assert_dynamic(mut state: glrmask::DynamicConstraintState<'_>) {
    assert!(allowed(&state.mask(), 0));
    assert!(!allowed(&state.mask(), 64));
    state.commit_token(0).unwrap();
    assert!(!state.is_accepting());
    assert!(allowed(&state.mask(), 64));
    state.commit_token(64).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn all_importers_support_grammar_level_end_tokens() {
    let grammar_vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let json_vocab = Vocab::new(vec![(0, b"\"a\"".to_vec())]);
    let json = r#"{"type":"string","enum":["a"]}"#;
    let ebnf = r#"start ::= "a""#;
    let lark = r#"start: "a""#;
    let glrm = "start start;\nt A ::= 'a';\nnt start ::= A;";
    let end_tokens = [64];
    let options = CompileOptions::default().end_tokens(&end_tokens);

    assert_static(Constraint::compile(Grammar::json_schema(json), &json_vocab, &options).unwrap().start());
    assert_static(Constraint::compile(Grammar::ebnf(ebnf), &grammar_vocab, &options).unwrap().start());
    assert_static(Constraint::compile(Grammar::lark(lark), &grammar_vocab, &options).unwrap().start());
    assert_static(Constraint::compile(Grammar::glrm(glrm), &grammar_vocab, &options).unwrap().start());

    assert_dynamic(DynamicConstraint::compile(Grammar::json_schema(json), &json_vocab, &options).unwrap().start());
    assert_dynamic(DynamicConstraint::compile(Grammar::ebnf(ebnf), &grammar_vocab, &options).unwrap().start());
    assert_dynamic(DynamicConstraint::compile(Grammar::lark(lark), &grammar_vocab, &options).unwrap().start());
    assert_dynamic(DynamicConstraint::compile(Grammar::glrm(glrm), &grammar_vocab, &options).unwrap().start());
}

#[test]
fn end_token_can_also_keep_byte_semantics() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (64, b"z".to_vec())]);
    let end_tokens = [64];
    let options = CompileOptions::default().end_tokens(&end_tokens);
    let constraint = Constraint::compile(Grammar::ebnf(r#"start ::= "a""#), &vocab, &options).unwrap();
    let mut state = constraint.start();
    assert!(!allowed(&state.mask(), 64));
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 64));
    state.commit_token(64).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn static_and_dynamic_state_rejection_semantics_match() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let grammar = Grammar::ebnf(r#"start ::= "a""#);
    let options = CompileOptions::default();

    let constraint = Constraint::compile(grammar, &vocab, &options).unwrap();
    let dynamic = DynamicConstraint::compile(grammar, &vocab, &options).unwrap();

    let mut static_state = constraint.start();
    let mut dynamic_state = dynamic.start();
    assert!(static_state.commit_token(99).is_err());
    assert!(dynamic_state.commit_token(99).is_err());
    assert!(!static_state.is_rejected());
    assert!(!dynamic_state.is_rejected());

    assert!(static_state.commit_token(1).is_err());
    assert!(dynamic_state.commit_token(1).is_err());
    assert!(static_state.is_rejected());
    assert!(dynamic_state.is_rejected());
}
