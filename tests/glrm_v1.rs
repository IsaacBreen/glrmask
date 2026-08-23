use glrmask::{
    CompileOptions, ConstraintSpec, DynamicConstraint, Grammar, StaticConstraint, Vocab,
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
extern token SPECIAL;
nt start = "a" SPECIAL "b";
"#;

#[test]
fn named_external_token_is_exact_not_a_byte_language() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (7, b"SPECIAL".to_vec())]);
    let spec = ConstraintSpec::builder(Grammar::glrm(EXACT_GRAMMAR), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [7])
        .unwrap()
        .build()
        .unwrap();
    let constraint = spec.compile_static(&CompileOptions::default()).unwrap();

    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 7));
    assert!(!allowed(&state.mask(), 1));

    let mut by_bytes = state.clone();
    assert!(by_bytes.commit_bytes(b"SPECIAL").is_err());

    state.commit_token(7).unwrap();
    assert!(allowed(&state.mask(), 1));
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn byte_less_decoder_ids_are_valid_and_multiple_ids_are_interchangeable() {
    let vocab = Vocab::new(Vec::new());
    let source = "glrm 1; start start; extern token SPECIAL; nt start = SPECIAL;";
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [100, 101])
        .unwrap()
        .build()
        .unwrap();
    let static_constraint = spec.compile_static(&CompileOptions::default()).unwrap();
    let dynamic_constraint = spec.compile_dynamic(&CompileOptions::default()).unwrap();

    for token in [100, 101] {
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
fn external_token_validation_is_early_and_complete() {
    let vocab = Vocab::new(vec![(7, Vec::new())]);
    let source = r#"
glrm 1;
start start;
extern token CONTROL;
extern grammar payload;
nt start = CONTROL payload;
"#;

    assert!(ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("CONTROL", [])
        .is_err());
    assert!(ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("payload", [7])
        .is_err());
    assert!(ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("CONTROL", [7])
        .unwrap()
        .build()
        .is_err());
}

#[test]
fn commit_token_keeps_exact_and_decoded_byte_interpretations() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"z".to_vec()), (7, b"a".to_vec())]);
    let source = r#"
glrm 1;
start start;
extern token SPECIAL;
nt start = "a" SPECIAL | "x" "a" "z";
"#;
    let constraint = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [7])
        .unwrap()
        .build()
        .unwrap()
        .compile_static(&CompileOptions::default())
        .unwrap();

    let mut byte_path = constraint.start();
    byte_path.commit_token(0).unwrap();
    byte_path.commit_token(7).unwrap();
    byte_path.commit_token(1).unwrap();
    assert!(byte_path.is_accepting());

    let mut mixed_path = constraint.start();
    mixed_path.commit_token(7).unwrap();
    mixed_path.commit_token(7).unwrap();
    assert!(mixed_path.is_accepting());
}

#[test]
fn exact_tokens_survive_static_and_dynamic_serialization() {
    let vocab = Vocab::new(vec![(100, Vec::new())]);
    let source = "glrm 1; start start; extern token SPECIAL; nt start = SPECIAL;";
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [100])
        .unwrap()
        .build()
        .unwrap();

    let compiled = spec.compile_static(&CompileOptions::default()).unwrap();
    let loaded = StaticConstraint::load(&compiled.save()).unwrap();
    let mut state = loaded.start();
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());

    let compiled = spec.compile_dynamic(&CompileOptions::default()).unwrap();
    let loaded = DynamicConstraint::load(&compiled.save()).unwrap();
    let mut state = loaded.start();
    state.commit_token(100).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn compiled_children_are_authoritative_and_compose_both_directions() {
    let vocab = Vocab::new(vec![(0, Vec::new()), (55, Vec::new()), (56, Vec::new())]);
    let child_source = "glrm 1; start start; extern token VALUE; nt start = VALUE;";
    let child_spec = ConstraintSpec::builder(Grammar::glrm(child_source), &vocab)
        .unwrap()
        .bind_token("VALUE", [55, 56])
        .unwrap()
        .build()
        .unwrap();
    let static_child = child_spec.compile_static(&CompileOptions::default()).unwrap();
    let dynamic_child = child_spec.compile_dynamic(&CompileOptions::default()).unwrap();
    let loaded_dynamic_child = DynamicConstraint::load(&dynamic_child.save()).unwrap();
    let parent_source = r#"
glrm 1;
start document;
extern token MARK;
extern grammar payload;
nt document = MARK payload;
"#;

    for (static_parent, dynamic_parent) in [
        {
            let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
                .unwrap()
                .bind_token("MARK", [0])
                .unwrap()
                .bind_grammar("payload", &static_child)
                .unwrap()
                .build()
                .unwrap();
            (
                spec.compile_static(&CompileOptions::default()).unwrap(),
                spec.compile_dynamic(&CompileOptions::default()).unwrap(),
            )
        },
        {
            let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
                .unwrap()
                .bind_token("MARK", [0])
                .unwrap()
                .bind_grammar("payload", &dynamic_child)
                .unwrap()
                .build()
                .unwrap();
            (
                spec.compile_static(&CompileOptions::default()).unwrap(),
                spec.compile_dynamic(&CompileOptions::default()).unwrap(),
            )
        },
        {
            let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
                .unwrap()
                .bind_token("MARK", [0])
                .unwrap()
                .bind_grammar("payload", &loaded_dynamic_child)
                .unwrap()
                .build()
                .unwrap();
            (
                spec.compile_static(&CompileOptions::default()).unwrap(),
                spec.compile_dynamic(&CompileOptions::default()).unwrap(),
            )
        },
    ] {
        for token in [55, 56] {
            let mut static_state = static_parent.start();
            let mut dynamic_state = dynamic_parent.start();
            static_state.commit_token(0).unwrap();
            dynamic_state.commit_token(0).unwrap();
            static_state.commit_token(token).unwrap();
            dynamic_state.commit_token(token).unwrap();
            assert!(static_state.is_accepting());
            assert!(dynamic_state.is_accepting());
        }
    }
}

#[test]
fn loaded_direct_regular_dynamic_child_remains_composable() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
    let dynamic_child = DynamicConstraint::compile(
        Grammar::lark("start: /a+/"),
        &vocab,
        &CompileOptions::default(),
    )
    .unwrap();
    let loaded_child = DynamicConstraint::load(&dynamic_child.save()).unwrap();
    let parent_source = "glrm 1; start start; extern grammar payload; nt start = payload;";
    let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
        .unwrap()
        .bind_grammar("payload", &loaded_child)
        .unwrap()
        .build()
        .unwrap();
    let static_parent = spec.compile_static(&CompileOptions::default()).unwrap();
    let dynamic_parent = spec.compile_dynamic(&CompileOptions::default()).unwrap();
    for token in [0, 1] {
        let mut static_state = static_parent.start();
        let mut dynamic_state = dynamic_parent.start();
        static_state.commit_token(token).unwrap();
        dynamic_state.commit_token(token).unwrap();
        assert!(static_state.is_accepting());
        assert!(dynamic_state.is_accepting());
    }
}

#[test]
fn grammar_can_bind_source_subgrammar_before_target_selection() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
    let parent = Grammar::glrm(
        "glrm 1; start start; extern grammar payload; nt start = payload;",
    )
    .bind_grammar("payload", Grammar::ebnf(r#"start ::= "x" | "y""#))
    .unwrap();

    let static_constraint =
        StaticConstraint::compile(parent.clone(), &vocab, &CompileOptions::default()).unwrap();
    let dynamic_constraint =
        DynamicConstraint::compile(parent, &vocab, &CompileOptions::default()).unwrap();

    for token in [0, 1] {
        let mut static_state = static_constraint.start();
        let mut dynamic_state = dynamic_constraint.start();
        static_state.commit_token(token).unwrap();
        dynamic_state.commit_token(token).unwrap();
        assert!(static_state.is_accepting());
        assert!(dynamic_state.is_accepting());
    }
}

#[test]
fn grammar_source_bindings_can_nest_and_mix_with_constraintspec_token_bindings() {
    let vocab = Vocab::new(vec![(7, Vec::new()), (8, b"z".to_vec())]);
    let leaf = Grammar::ebnf(r#"start ::= "z""#);
    let child = Grammar::glrm(
        "glrm 1; start start; extern grammar leaf; nt start = leaf;",
    )
    .bind_grammar("leaf", leaf)
    .unwrap();
    let parent = Grammar::glrm(
        "glrm 1; start start; extern token MARK; extern grammar child; nt start = MARK child;",
    )
    .bind_grammar("child", child)
    .unwrap();

    let spec = ConstraintSpec::builder(parent, &vocab)
        .unwrap()
        .bind_token("MARK", [7])
        .unwrap()
        .build()
        .unwrap();
    let constraint = spec.compile_static(&CompileOptions::default()).unwrap();
    let mut state = constraint.start();
    state.commit_token(7).unwrap();
    state.commit_token(8).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn grammar_source_binding_accepts_shorter_lived_child_source() {
    let child_source = String::from(r#"start ::= "x""#);
    let _grammar = Grammar::glrm(
        "glrm 1; start start; extern grammar payload; nt start = payload;",
    )
    .bind_grammar("payload", Grammar::ebnf(&child_source))
    .unwrap();
}

#[test]
fn grammar_source_bindings_validate_parent_declarations() {
    let wrong_kind = Grammar::glrm(
        "glrm 1; start start; extern token X; nt start = X;",
    )
    .bind_grammar("X", Grammar::ebnf(r#"start ::= "x""#))
    .unwrap_err()
    .to_string();
    assert!(wrong_kind.contains("kind token, not grammar"), "{wrong_kind}");

    let unknown = Grammar::glrm("glrm 1; start start; nt start = \"x\";")
        .bind_grammar("missing", Grammar::ebnf(r#"start ::= "x""#))
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("no external grammar"), "{unknown}");
}

#[test]
fn bind_grammar_accepts_source_and_spec_and_does_not_inherit_parent_bindings() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec())]);
    let parent = "glrm 1; start start; extern grammar child; nt start = child;";
    let source_child = Grammar::ebnf(r#"start ::= "x""#);
    let source_bound = ConstraintSpec::builder(Grammar::glrm(parent), &vocab)
        .unwrap()
        .bind_grammar("child", source_child.clone())
        .unwrap()
        .build()
        .unwrap()
        .compile_static(&CompileOptions::default())
        .unwrap();
    let mut state = source_bound.start();
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());

    let child_spec = ConstraintSpec::builder(source_child, &vocab)
        .unwrap()
        .build()
        .unwrap();
    assert!(ConstraintSpec::builder(Grammar::glrm(parent), &vocab)
        .unwrap()
        .bind_grammar("child", child_spec)
        .unwrap()
        .build()
        .is_ok());

    let unresolved_child = Grammar::glrm(
        "glrm 1; start start; extern token TOKEN; nt start = TOKEN;",
    );
    let spec = ConstraintSpec::builder(Grammar::glrm(parent), &vocab)
        .unwrap()
        .bind_grammar("child", unresolved_child)
        .unwrap()
        .build()
        .unwrap();
    assert!(spec.compile_static(&CompileOptions::default()).is_err());
}

#[test]
fn incompatible_compiled_child_target_is_rejected_at_bind_time() {
    let child_vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let parent_vocab = Vocab::new(vec![(0, b"b".to_vec())]);
    let child = StaticConstraint::compile(
        Grammar::ebnf(r#"start ::= "a""#),
        &child_vocab,
        &CompileOptions::default(),
    )
    .unwrap();
    let parent = "glrm 1; start start; extern grammar child; nt start = child;";
    assert!(ConstraintSpec::builder(Grammar::glrm(parent), &parent_vocab)
        .unwrap()
        .bind_grammar("child", &child)
        .is_err());
}

#[test]
fn v1_terminal_fa_compiles_and_masks_tokens() {
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
    let constraint = StaticConstraint::compile(
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
fn nested_external_tokens_use_qualified_names() {
    let vocab = Vocab::new(vec![(123, Vec::new())]);
    let source = r#"
glrm 1;
start document;
g inner = {
    start value;
    extern token END;
    nt value = END;
};
nt document = inner;
"#;
    let constraint = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("inner::END", [123])
        .unwrap()
        .build()
        .unwrap()
        .compile_static(&CompileOptions::default())
        .unwrap();
    let mut state = constraint.start();
    state.commit_token(123).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn external_token_can_label_nonterminal_fa_edges() {
    let vocab = Vocab::new(vec![(77, Vec::new())]);
    let source = r#"
glrm 1;
start start;
extern token END;
nt start = fa {
    start begin;
    accept done;
    begin -> done: END;
};
"#;
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("END", [77])
        .unwrap()
        .build()
        .unwrap();
    assert!(allowed(&spec.compile_static(&CompileOptions::default()).unwrap().start().mask(), 77));
    assert!(allowed(&spec.compile_dynamic(&CompileOptions::default()).unwrap().start().mask(), 77));
}

#[test]
fn lexer_pragma_does_not_change_token_language() {
    let vocab = Vocab::new(vec![(0, b"abc".to_vec()), (1, b"a".to_vec()), (2, b"bc".to_vec())]);
    let plain = "glrm 1; start start; t WORD = /[a-z]+/; nt start = WORD;";
    let hinted = r#"
glrm 1;
start start;
pragma glrmask { lexer group words = WORD; }
t WORD = /[a-z]+/;
nt start = WORD;
"#;
    let options = CompileOptions::default();
    let plain = StaticConstraint::compile(Grammar::glrm(plain), &vocab, &options).unwrap();
    let hinted = StaticConstraint::compile(Grammar::glrm(hinted), &vocab, &options).unwrap();
    let mut left = plain.start();
    let mut right = hinted.start();
    assert_eq!(left.mask(), right.mask());
    left.commit_token(1).unwrap();
    right.commit_token(1).unwrap();
    assert_eq!(left.mask(), right.mask());
}
