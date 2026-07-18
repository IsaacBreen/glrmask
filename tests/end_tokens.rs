use glrmask::{Constraint, DynamicConstraint, Vocab};

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
    assert!(!state.is_complete());
    assert!(allowed(&state.mask(), 64));
    state.commit_token(64).unwrap();
    assert!(state.is_complete());
}

fn assert_dynamic(mut state: glrmask::DynamicConstraintState<'_>) {
    assert!(allowed(&state.mask(), 0));
    assert!(!allowed(&state.mask(), 64));
    state.commit_token(0).unwrap();
    assert!(!state.is_complete());
    assert!(allowed(&state.mask(), 64));
    state.commit_token(64).unwrap();
    assert!(state.is_complete());
}

#[test]
fn all_importers_support_grammar_level_end_tokens() {
    let grammar_vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let json_vocab = Vocab::new(vec![(0, b"\"a\"".to_vec())]);
    let json = r#"{"type":"string","enum":["a"]}"#;
    let ebnf = r#"start ::= "a""#;
    let lark = r#"start: "a""#;
    let glrm = "start start;\nt A ::= 'a';\nnt start ::= A;";

    assert_static(Constraint::from_json_schema_with_end_tokens(json, &json_vocab, &[64]).unwrap().start());
    assert_static(Constraint::from_ebnf_with_end_tokens(ebnf, &grammar_vocab, &[64]).unwrap().start());
    assert_static(Constraint::from_lark_with_end_tokens(lark, &grammar_vocab, &[64]).unwrap().start());
    assert_static(Constraint::from_glrm_grammar_with_end_tokens(glrm, &grammar_vocab, &[64]).unwrap().start());

    assert_dynamic(DynamicConstraint::from_json_schema_with_end_tokens(json, &json_vocab, &[64]).unwrap().start());
    assert_dynamic(DynamicConstraint::from_ebnf_with_end_tokens(ebnf, &grammar_vocab, &[64]).unwrap().start());
    assert_dynamic(DynamicConstraint::from_lark_with_end_tokens(lark, &grammar_vocab, &[64]).unwrap().start());
    assert_dynamic(DynamicConstraint::from_glrm_grammar_with_end_tokens(glrm, &grammar_vocab, &[64]).unwrap().start());
}

#[test]
fn end_token_can_also_keep_byte_semantics() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (64, b"z".to_vec())]);
    let constraint = Constraint::from_ebnf_with_end_tokens(r#"start ::= "a""#, &vocab, &[64]).unwrap();
    let mut state = constraint.start();
    assert!(!allowed(&state.mask(), 64));
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 64));
    state.commit_token(64).unwrap();
    assert!(state.is_complete());
}
