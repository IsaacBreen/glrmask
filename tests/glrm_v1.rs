use glrmask::{
    CompileOptions, Constraint, DynamicConstraint, ExternalTerminalBinding, Grammar, Vocab,
};

fn allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word)
        .is_some_and(|value| value & (1u32 << bit) != 0)
}

const EXACT_GRAMMAR: &str = r#"
glrm 1;
start start;
extern t SPECIAL;
nt start = "a" SPECIAL "b";
"#;

#[test]
fn named_external_terminal_is_an_exact_token_not_a_byte_language() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (7, b"SPECIAL".to_vec())]);
    let ids = [7];
    let bindings = [ExternalTerminalBinding::new("SPECIAL", &ids)];
    let options = CompileOptions::default().external_terminal_bindings(&bindings);
    let constraint = Constraint::compile(Grammar::glrm(EXACT_GRAMMAR), &vocab, &options).unwrap();

    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
    assert_eq!(allowed(&state.mask(), 7), true);
    assert_eq!(allowed(&state.mask(), 1), false);

    let mut by_bytes = state.clone();
    assert!(by_bytes.commit_bytes(b"SPECIAL").is_err());

    state.commit_token(7).unwrap();
    assert!(allowed(&state.mask(), 1));
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());
    assert!(!state.is_rejected());
}

#[test]
fn external_terminal_ids_can_be_outside_the_byte_vocab_and_multiple_ids_are_interchangeable() {
    let vocab = Vocab::new(Vec::new());
    let ids = [100, 101];
    let bindings = [ExternalTerminalBinding::new("SPECIAL", &ids)];
    let options = CompileOptions::default().external_terminal_bindings(&bindings);
    let source = "glrm 1; start start; extern t SPECIAL; nt start = SPECIAL;";

    let static_constraint = Constraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    let dynamic_constraint = DynamicConstraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();

    for token in ids {
        let mut static_state = static_constraint.start();
        let mut dynamic_state = dynamic_constraint.start();
        assert!(allowed(&static_state.mask(), 100));
        assert!(allowed(&static_state.mask(), 101));
        assert_eq!(static_state.mask(), dynamic_state.mask());
        static_state.commit_token(token).unwrap();
        dynamic_state.commit_token(token).unwrap();
        assert!(static_state.is_accepting());
        assert!(dynamic_state.is_accepting());
    }
}

#[test]
fn externally_bound_id_keeps_ordinary_byte_semantics_on_other_parse_paths() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"z".to_vec()),
        (7, b"a".to_vec()),
    ]);
    let ids = [7];
    let bindings = [ExternalTerminalBinding::new("SPECIAL", &ids)];
    let options = CompileOptions::default().external_terminal_bindings(&bindings);
    let source = r#"
glrm 1;
start start;
extern t SPECIAL;
nt start = "a" SPECIAL | "x" "a" "z";
"#;
    let constraint = Constraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();

    // Token 7 has bytes "a", so it follows the ordinary byte path here.
    let mut byte_path = constraint.start();
    byte_path.commit_token(0).unwrap();
    assert!(allowed(&byte_path.mask(), 7));
    byte_path.commit_token(7).unwrap();
    assert!(allowed(&byte_path.mask(), 1));
    byte_path.commit_token(1).unwrap();
    assert!(byte_path.is_accepting());

    // At the root the same token's byte realization starts the first branch;
    // the next occurrence is admitted as the exact SPECIAL terminal.
    let mut mixed_path = constraint.start();
    mixed_path.commit_token(7).unwrap();
    assert!(allowed(&mixed_path.mask(), 7));
    mixed_path.commit_token(7).unwrap();
    assert!(mixed_path.is_accepting());
}

#[test]
fn end_tokens_remain_separate_from_named_external_terminals() {
    let vocab = Vocab::new(Vec::new());
    let special_ids = [100];
    let end_ids = [64];
    let bindings = [ExternalTerminalBinding::new("SPECIAL", &special_ids)];
    let options = CompileOptions::default()
        .external_terminal_bindings(&bindings)
        .end_token_ids(&end_ids);
    let source = "glrm 1; start start; extern t SPECIAL; nt start = SPECIAL;";

    for mut state in [
        Constraint::compile(Grammar::glrm(source), &vocab, &options)
            .unwrap()
            .start(),
    ] {
        assert!(allowed(&state.mask(), 100));
        assert!(!allowed(&state.mask(), 64));
        state.commit_token(100).unwrap();
        assert!(!state.is_accepting());
        assert!(allowed(&state.mask(), 64));
        state.commit_token(64).unwrap();
        assert!(state.is_accepting());
    }

    let dynamic = DynamicConstraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    let mut state = dynamic.start();
    state.commit_token(100).unwrap();
    assert!(!state.is_accepting());
    assert!(allowed(&state.mask(), 64));
    state.commit_token(64).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn named_external_terminals_survive_static_and_dynamic_serialization() {
    let vocab = Vocab::new(Vec::new());
    let ids = [100];
    let bindings = [ExternalTerminalBinding::new("SPECIAL", &ids)];
    let options = CompileOptions::default().external_terminal_bindings(&bindings);
    let source = "glrm 1; start start; extern t SPECIAL; nt start = SPECIAL;";

    let static_constraint = Constraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    let loaded = Constraint::load(&static_constraint.save()).unwrap();
    let mut state = loaded.start();
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());

    let dynamic = DynamicConstraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
    let mut state = loaded.start();
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn named_external_terminals_compose_with_external_subgrammars_without_sentinel_collisions() {
    let vocab = Vocab::new(Vec::new());
    let child = Constraint::compile(
        Grammar::glrm("start start; nt start ::= @token(55);"),
        &vocab,
        &CompileOptions::default(),
    )
    .unwrap();

    // ID 0 is exactly where an empty-vocab linker allocator would otherwise
    // begin. The compiler must reserve it for MARK and choose another hidden ID.
    let mark_ids = [0];
    let bindings = [ExternalTerminalBinding::new("MARK", &mark_ids)];
    let children = [("payload", &child)];
    let options = CompileOptions::default()
        .external_terminal_bindings(&bindings)
        .subgrammars(&children);
    let source = r#"
glrm 1;
start document;
extern t MARK;
extern g payload;
nt document = MARK payload;
"#;

    let constraint = Constraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 55));
    state.commit_token(55).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn v1_terminal_fa_compiles_directly_and_masks_tokens() {
    let vocab = Vocab::new(vec![(0, b"ab".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec())]);
    let source = r#"
glrm 1;
start start;
t WORD = fa {
    start begin;
    accept done;
    begin -> middle: "a";
    middle -> done: "b";
};
nt start = WORD;
"#;
    let constraint = Constraint::compile(
        Grammar::glrm(source),
        &vocab,
        &CompileOptions::default(),
    )
    .unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    assert!(allowed(&state.mask(), 1));
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn nested_external_terminal_bindings_use_qualified_names() {
    let vocab = Vocab::new(Vec::new());
    let ids = [123];
    let bindings = [ExternalTerminalBinding::new("inner::END", &ids)];
    let options = CompileOptions::default().external_terminal_bindings(&bindings);
    let source = r#"
glrm 1;
start document;
g inner = {
    start value;
    extern t END;
    nt value = END;
};
nt document = inner;
"#;
    let constraint = Constraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 123));
    state.commit_token(123).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn external_terminal_can_label_nonterminal_fa_edges() {
    let vocab = Vocab::new(Vec::new());
    let ids = [77];
    let bindings = [ExternalTerminalBinding::new("END", &ids)];
    let options = CompileOptions::default().external_terminal_bindings(&bindings);
    let source = r#"
glrm 1;
start start;
extern t END;
nt start = fa {
    start begin;
    accept done;
    begin -> done: END;
};
"#;
    let constraint = Constraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    let dynamic = DynamicConstraint::compile(Grammar::glrm(source), &vocab, &options).unwrap();
    assert!(allowed(&constraint.start().mask(), 77));
    assert!(allowed(&dynamic.start().mask(), 77));
}

#[test]
fn glrmask_lexer_pragma_does_not_change_token_language() {
    let vocab = Vocab::new(vec![
        (0, b"abc".to_vec()),
        (1, b"a".to_vec()),
        (2, b"bc".to_vec()),
        (3, b"123".to_vec()),
    ]);
    let plain = r#"
glrm 1;
start start;
t WORD = /[a-z]+/;
nt start = WORD;
"#;
    let hinted = r#"
glrm 1;
start start;
pragma glrmask {
    lexer group words = WORD;
}
t WORD = /[a-z]+/;
nt start = WORD;
"#;
    let options = CompileOptions::default();
    let plain = Constraint::compile(Grammar::glrm(plain), &vocab, &options).unwrap();
    let hinted = Constraint::compile(Grammar::glrm(hinted), &vocab, &options).unwrap();

    let mut left = plain.start();
    let mut right = hinted.start();
    assert_eq!(left.mask(), right.mask());
    left.commit_token(1).unwrap();
    right.commit_token(1).unwrap();
    assert_eq!(left.mask(), right.mask());
    left.commit_token(2).unwrap();
    right.commit_token(2).unwrap();
    assert_eq!(left.is_accepting(), right.is_accepting());
}

#[test]
fn static_and_dynamic_reject_external_bindings_for_non_glrm_grammars() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let ids = [7];
    let bindings = [ExternalTerminalBinding::new("SPECIAL", &ids)];
    let options = CompileOptions::default().external_terminal_bindings(&bindings);
    for grammar in [
        Grammar::ebnf(r#"start ::= "a""#),
        Grammar::lark(r#"start: "a""#),
        Grammar::json_schema(r#"{"type":"string"}"#),
    ] {
        assert!(Constraint::compile(grammar, &vocab, &options).is_err());
        assert!(DynamicConstraint::compile(grammar, &vocab, &options).is_err());
    }
}
