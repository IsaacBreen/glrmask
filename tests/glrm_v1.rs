use glrmask::test_modes::MaskMode;
use glrmask::{
    Constraint, ConstraintSpec, DynamicConstraint, Grammar, PreparedGrammar,
    Vocab};

// Composition fixtures explicitly materialize the native parser before binding.
// The ordinary Dynamic default remains LR-backed; mixed native/LR rejection is checked.
fn compile_native_dynamic_fixture(grammar: Grammar, vocab: &Vocab) -> glrmask::Result<DynamicConstraint> {
    glrmask::__private::into_dynamic_template_parser(DynamicConstraint::compile(grammar, vocab)?)
}

fn assert_lr_composition_rejected<T>(result: glrmask::Result<T>) {
    match result {
        Err(error) if matches!(error.kind(), glrmask::ErrorKind::Compilation(_)) => {
            let glrmask::ErrorKind::Compilation(message) = error.kind() else { unreachable!() };
            assert!(
                message.contains("a composition can't mix parts precompiled with the LrTable and TemplateDfa parser backends"), "{message}");
        }
        Err(error) => panic!("unexpected composition error: {error}"),
        Ok(_) => panic!("LR-backed composition unexpectedly succeeded"),
    }
}

fn allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word)
        .is_some_and(|value| value & (1u32 << bit) != 0)
}

fn assert_static_xy_matches(reference: &Constraint, candidate: &Constraint) {
    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    assert_eq!(candidate_state.mask(), reference_state.mask());
    assert!(allowed(&candidate_state.mask(), 2));
    assert!(!allowed(&candidate_state.mask(), 3));
    assert!(candidate_state.commit_token(2));
    assert!(reference_state.commit_token(2));
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());

    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    assert!(candidate_state.commit_token(0));
    assert!(reference_state.commit_token(0));
    assert_eq!(candidate_state.mask(), reference_state.mask());
    assert!(candidate_state.commit_token(1));
    assert!(reference_state.commit_token(1));
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());
}

fn assert_dynamic_xy_matches(reference: &Constraint, candidate: &DynamicConstraint) {
    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    assert_eq!(candidate_state.mask(), reference_state.mask());
    assert!(allowed(&candidate_state.mask(), 2));
    assert!(!allowed(&candidate_state.mask(), 3));
    assert!(candidate_state.commit_token(2));
    assert!(reference_state.commit_token(2));
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());

    let mut reference_state = reference.start();
    let mut candidate_state = candidate.start();
    assert!(candidate_state.commit_token(0));
    assert!(reference_state.commit_token(0));
    assert_eq!(candidate_state.mask(), reference_state.mask());
    assert!(candidate_state.commit_token(1));
    assert!(reference_state.commit_token(1));
    assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());
}

fn assert_static_xyz_matches(reference: &Constraint, candidate: &Constraint) {
    for tokens in [&[5][..], &[2, 3][..], &[0, 4][..], &[0, 1, 3][..]] {
        let mut reference_state = reference.start();
        let mut candidate_state = candidate.start();
        for &token in tokens {
            assert_eq!(candidate_state.mask(), reference_state.mask());
            assert!(candidate_state.commit_token(token));
            assert!(reference_state.commit_token(token));
        }
        assert_eq!(candidate_state.is_accepting(), reference_state.is_accepting());
    }
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
    let constraint = spec.compile().unwrap();

    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    assert!(state.commit_token(0));
    assert!(allowed(&state.mask(), 7));
    assert!(!allowed(&state.mask(), 1));

    let mut by_bytes = state.clone();
    assert!(!by_bytes.commit_bytes(b"SPECIAL"));

    assert!(state.commit_token(7));
    assert!(allowed(&state.mask(), 1));
    assert!(state.commit_token(1));
    assert!(state.is_accepting());
}

#[test]
fn byte_less_decoder_ids_are_valid_and_multiple_ids_are_interchangeable() {
    let vocab = Vocab::new(Vec::new()).with_special_tokens([100,101]);
    let source = "glrm 1; start start; extern token SPECIAL; nt start = SPECIAL;";
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("SPECIAL", [100, 101])
        .unwrap()
        .build()
        .unwrap();
    let static_constraint = spec.compile().unwrap();
    let dynamic_constraint = spec.compile_dynamic().unwrap();

    for token in [100, 101] {
        let mut static_state = static_constraint.start();
        let mut dynamic_state = dynamic_constraint.start();
        assert!(allowed(&static_state.mask(), 100));
        assert!(allowed(&static_state.mask(), 101));
        assert_eq!(static_state.mask(), dynamic_state.mask());
        assert!(static_state.commit_token(token));
        assert!(dynamic_state.commit_token(token));
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

    let empty = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("CONTROL", [])
        .unwrap().build().unwrap().compile().unwrap();
    assert!(!empty.start().is_accepting());
    assert!(empty.start().mask().iter().all(|word| *word == 0));
    // The legacy/internal explicit bind_token helper remains narrowly typed.
    // Public Grammar::bind performs the intentional event-to-grammar widening.
    let narrow = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("CONTROL", [7]).unwrap()
        .bind_token("payload", [7]).unwrap()
        .build().unwrap();
    assert!(narrow.compile().unwrap_err().to_string().contains("is a grammar extern"));
    let mut widened = Grammar::from_glrm(source).unwrap();
    widened.bind("CONTROL", glrmask::TokenSet::from(7)).unwrap();
    widened.bind("payload", glrmask::TokenSet::from(7)).unwrap();
    for mode in MaskMode::ALL {
        let constraint = widened.compile_with_build_options(
            &vocab, mode.build_options()).unwrap();
        let mut state = constraint.start();
        for _ in 0..2 {
            assert!(!state.is_accepting());
            assert!(allowed(&state.mask(), 7));
            assert!(state.commit_token(7));
        }
        assert!(state.is_accepting());
    }
    // External references may remain unresolved until final compilation or link.
    let open_spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("CONTROL", [7])
        .unwrap()
        .build()
        .unwrap();
    let static_open = open_spec.clone().compile().unwrap();
    let dynamic_open = open_spec.compile_dynamic().unwrap();
    let mut static_state = static_open.start();
    let mut dynamic_state = dynamic_open.start();
    assert_eq!(static_state.mask(), dynamic_state.mask());
    assert!(allowed(&static_state.mask(), 7));
    assert!(static_state.commit_token(7));
    assert!(dynamic_state.commit_token(7));
    assert!(!static_state.is_accepting());
    assert!(!dynamic_state.is_accepting());
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
        .compile()
        .unwrap();

    let mut byte_path = constraint.start();
    assert!(byte_path.commit_token(0));
    assert!(byte_path.commit_token(7));
    assert!(byte_path.commit_token(1));
    assert!(byte_path.is_accepting());

    let mut mixed_path = constraint.start();
    assert!(mixed_path.commit_token(7));
    assert!(mixed_path.commit_token(7));
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

    let compiled = spec.compile().unwrap();
    let loaded = Constraint::load(&compiled.save(), &vocab).unwrap();
    let mut state = loaded.start();
    assert!(state.commit_token(100));
    assert!(state.is_accepting());

    let compiled = spec.compile_dynamic().unwrap();
    let loaded = DynamicConstraint::load(&compiled.save(), &vocab).unwrap();
    let mut state = loaded.start();
    assert!(state.commit_token(100));
    assert!(state.is_accepting());
}

#[test]
fn compiled_parent_late_binding_matches_monolithic_across_backend_matrix() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"y".to_vec()),
        (2, b"xy".to_vec()),
        // Relevant prefix followed by a suffix outside the language.
        (3, b"xyz".to_vec()),
    ]);
    let parent_source =
        "glrm 1; start start; extern grammar child; nt start = \"x\" child;";
    let child_grammar = Grammar::ebnf(r#"start ::= "y""#);
    let reference = Constraint::compile(Grammar::ebnf(r#"start ::= "x" "y""#), &vocab)
        .unwrap();
    let static_child = Constraint::compile(child_grammar.clone(), &vocab).unwrap();
    let prepared_child = child_grammar.prepare(&vocab).unwrap();
    let dynamic_child = compile_native_dynamic_fixture(child_grammar, &vocab).unwrap();
    let static_parent = Constraint::compile(Grammar::glrm(parent_source), &vocab).unwrap();
    let dynamic_parent =
        compile_native_dynamic_fixture(Grammar::glrm(parent_source), &vocab).unwrap();

    let lr_child = DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
    let lr_parent = DynamicConstraint::compile(Grammar::glrm(parent_source), &vocab).unwrap();
    assert_lr_composition_rejected(static_parent.bind_grammar("child", &lr_child));
    assert_lr_composition_rejected(lr_parent.bind_grammar("child", &static_child));
    assert_lr_composition_rejected(lr_parent.bind_grammar_dynamic_boundary("child", &dynamic_child));

    assert_static_xy_matches(
        &reference,
        &static_parent.bind_grammar("child", &static_child).unwrap(),
    );
    assert_static_xy_matches(
        &reference,
        &static_parent.bind_grammar("child", &dynamic_child).unwrap(),
    );
    assert_static_xy_matches(
        &reference,
        &static_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap(),
    );
    assert_static_xy_matches(
        &reference,
        &static_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap(),
    );

    assert_dynamic_xy_matches(
        &reference,
        &dynamic_parent.bind_grammar("child", &static_child).unwrap(),
    );
    assert_dynamic_xy_matches(
        &reference,
        &dynamic_parent.bind_grammar("child", &dynamic_child).unwrap(),
    );
    assert_dynamic_xy_matches(
        &reference,
        &dynamic_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap(),
    );
    let dynamic_bound_static_boundary = dynamic_parent.bind_grammar("child", &dynamic_child).unwrap();
    assert_dynamic_xy_matches(&reference, &dynamic_bound_static_boundary);
    let dynamic_bound_dynamic_boundary = dynamic_parent
        .bind_grammar_dynamic_boundary("child", &dynamic_child)
        .unwrap();
    assert_dynamic_xy_matches(&reference, &dynamic_bound_dynamic_boundary);

    let loaded_dynamic_bound_static =
        DynamicConstraint::load(&dynamic_bound_static_boundary.save(), &vocab).unwrap();
    assert_dynamic_xy_matches(&reference, &loaded_dynamic_bound_static);
    let loaded_dynamic_bound_dynamic =
        DynamicConstraint::load(&dynamic_bound_dynamic_boundary.save(), &vocab).unwrap();
    assert_dynamic_xy_matches(&reference, &loaded_dynamic_bound_dynamic);

    let loaded_static_parent = PreparedGrammar::load(
        Grammar::from_glrm(parent_source).unwrap()
            .prepare(&vocab)
            .unwrap()
            .save(),
        &vocab,
    )
    .unwrap();
    let loaded_static_bound = loaded_static_parent
        .bind_snapshot("child", &prepared_child)
        .unwrap()
        .link()
        .unwrap();
    assert_static_xy_matches(&reference, &loaded_static_bound);

    let error = DynamicConstraint::load(&dynamic_parent.save(), &vocab).unwrap_err();
    assert!(
        error.to_string().contains("unresolved slots"),
        "an unresolved parent must not load as a runnable constraint: {error}",
    );
    let loaded_dynamic_parent = PreparedGrammar::load(
        Grammar::from_glrm(parent_source).unwrap()
            .prepare(&vocab)
            .unwrap()
            .save(),
        &vocab,
    )
    .unwrap();
    let loaded_dynamic_bound = loaded_dynamic_parent
        .bind_snapshot("child", &prepared_child)
        .unwrap()
        .link_with(glrmask::test_modes::MaskMode::C.build_options())
        .unwrap();
    assert_static_xy_matches(&reference, &loaded_dynamic_bound);

    assert!(static_parent.bind_grammar("missing", &static_child).is_err());
    assert!(dynamic_parent
        .bind_grammar_dynamic_boundary("missing", &dynamic_child)
        .is_err());
}

#[test]
fn unresolved_late_parent_roundtrip_excludes_private_linker_token_from_masks() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"y".to_vec()),
        (2, b"xy".to_vec()),
    ]);
    let parent = Grammar::from_glrm(
        "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
    ).unwrap()
    .prepare(&vocab)
    .unwrap();

    // An unresolved grammar slot belongs to PreparedGrammar, never Constraint. Its
    // private linker coordinate survives PreparedGrammar persistence without becoming
    // part of the public model-token mask.
    let loaded = PreparedGrammar::load(parent.save(), &vocab).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "y""#).unwrap()
        .prepare(&vocab)
        .unwrap();
    let reference = Constraint::compile(Grammar::ebnf(r#"start ::= "x" "y""#), &vocab).unwrap();
    let bound = loaded
        .bind_snapshot("child", &child)
        .unwrap()
        .link()
        .unwrap();
    assert_eq!(bound.mask_len(), 1);
    assert_eq!(bound.start().mask().len(), 1);
    assert_static_xy_matches(&reference, &bound);
}

#[test]
fn late_binding_multiple_adjacent_slots_handles_internal_multi_boundary_tokens() {
    let vocab = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"y".to_vec()),
        (2, b"xy".to_vec()),
        (3, b"z".to_vec()),
        (4, b"yz".to_vec()),
        (5, b"xyz".to_vec()),
        (6, b"xyzz".to_vec()),
    ]);
    let parent = Grammar::glrm(
        "glrm 1; start start; extern grammar left; extern grammar right; \
         nt start = \"x\" left right;",
    );
    let reference =
        Constraint::compile(Grammar::ebnf(r#"start ::= "x" "y" "z""#), &vocab).unwrap();
    let static_left = Constraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
    let dynamic_right =
        compile_native_dynamic_fixture(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap();

    let open = Constraint::compile(parent, &vocab).unwrap();
    let partial = open.bind_grammar("left", &static_left).unwrap();
    let complete = partial
        .bind_grammar_dynamic_boundary("right", &dynamic_right)
        .unwrap();
    assert_static_xyz_matches(&reference, &complete);
    let state = complete.start();
    assert!(!allowed(&state.mask(), 6));

    let complete = partial
        .bind_grammar_dynamic_boundary("right", &dynamic_right)
        .unwrap();
    assert_static_xyz_matches(&reference, &complete);
    let state = complete.start();
    assert!(!allowed(&state.mask(), 6));
}

#[test]
fn dynamic_boundary_cross_prefilter_preserves_scoped_ignore_paths() {
    let vocab = Vocab::new(vec![
        (0, b"X".to_vec()),
        (1, b" ".to_vec()),
        (2, b"\t".to_vec()),
        (3, b"a".to_vec()),
        (4, b"!".to_vec()),
        (5, b"X a !".to_vec()),
        (6, b" \ta".to_vec()),
        (7, b"a\t ".to_vec()),
        (8, b"\t\t".to_vec()),
        (9, b"  ".to_vec()),
        (10, b"a ".to_vec()),
    ]);
    let parent = Constraint::compile(
        Grammar::glrm(
            r#"
                glrm 1;
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern grammar child;
                nt document = "X" child "!";
            "#,
        ),
        &vocab,
    )
    .unwrap();
    let child = compile_native_dynamic_fixture(
        Grammar::glrm(
            r#"
                glrm 1;
                start child;
                nt child = "a";
            "#,
        ),
        &vocab,
    )
    .unwrap();
    let reference = Constraint::compile(
        Grammar::glrm(
            r#"
                glrm 1;
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                g child = {
                    start child;
                    nt child = "a";
                };
                nt document = "X" child "!";
            "#,
        ),
        &vocab,
    )
    .unwrap();
    let bound = parent
        .bind_grammar_dynamic_boundary("child", &child)
        .unwrap();
    let loaded = Constraint::load(&bound.save(), &vocab).unwrap();

    for tokens in [
        &[5][..],
        &[0, 3, 4][..],
        &[0, 9, 3, 4][..],
        &[0, 10, 4][..],
    ] {
        let mut expected = reference.start();
        let mut actual = bound.start();
        let mut restored = loaded.start();
        for &token in tokens {
            assert_eq!(actual.mask(), expected.mask(), "bound mask before {tokens:?}");
            assert_eq!(restored.mask(), expected.mask(), "loaded mask before {tokens:?}");
            assert!(actual.commit_token(token));
            assert!(restored.commit_token(token));
            assert!(expected.commit_token(token));
        }
        assert_eq!(actual.is_accepting(), expected.is_accepting(), "bound {tokens:?}");
        assert_eq!(restored.is_accepting(), expected.is_accepting(), "loaded {tokens:?}");
    }
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
    let static_child = child_spec.compile().unwrap();
    let dynamic_child = glrmask::__private::into_dynamic_template_parser(child_spec.compile_dynamic().unwrap()).unwrap();
    let loaded_dynamic_child = DynamicConstraint::load(&dynamic_child.save(), &vocab).unwrap();
    let parent_source = r#"
glrm 1;
start document;
extern token MARK;
extern grammar payload;
nt document = MARK payload;
"#;

    let open_parent_spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
        .unwrap().bind_token("MARK", [0]).unwrap().build().unwrap();
    let native_open_parent = glrmask::__private::into_dynamic_template_parser(
        open_parent_spec.compile_dynamic().unwrap()).unwrap();

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
                spec.compile().unwrap(),
                native_open_parent.bind_grammar("payload", &static_child).unwrap(),
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
                spec.compile().unwrap(),
                native_open_parent.bind_grammar("payload", &dynamic_child).unwrap(),
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
                spec.compile().unwrap(),
                native_open_parent.bind_grammar("payload", &loaded_dynamic_child).unwrap(),
            )
        },
    ] {
        for token in [55, 56] {
            let mut static_state = static_parent.start();
            let mut dynamic_state = dynamic_parent.start();
            assert!(static_state.commit_token(0));
            assert!(dynamic_state.commit_token(0));
            assert!(static_state.commit_token(token));
            assert!(dynamic_state.commit_token(token));
            assert!(static_state.is_accepting());
            assert!(dynamic_state.is_accepting());
        }
    }
}

#[test]
fn loaded_direct_regular_dynamic_child_remains_composable() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
    let dynamic_child = compile_native_dynamic_fixture(
        Grammar::lark("start: /a+/"),
        &vocab,
    )
    .unwrap();
    let loaded_child = DynamicConstraint::load(&dynamic_child.save(), &vocab).unwrap();
    let parent_source = "glrm 1; start start; extern grammar payload; nt start = payload;";
    let spec = ConstraintSpec::builder(Grammar::glrm(parent_source), &vocab)
        .unwrap()
        .bind_grammar("payload", &loaded_child)
        .unwrap()
        .build()
        .unwrap();
    let static_parent = spec.compile().unwrap();
    let dynamic_parent = compile_native_dynamic_fixture(Grammar::glrm(parent_source), &vocab)
        .unwrap().bind_grammar("payload", &loaded_child).unwrap();
    for token in [0, 1] {
        let mut static_state = static_parent.start();
        let mut dynamic_state = dynamic_parent.start();
        assert!(static_state.commit_token(token));
        assert!(dynamic_state.commit_token(token));
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
        Constraint::compile(parent.clone(), &vocab).unwrap();
    assert_lr_composition_rejected(DynamicConstraint::compile(parent.clone(), &vocab));
    let dynamic_constraint =
        parent.compile_with_build_options(&vocab, glrmask::test_modes::MaskMode::C.build_options()).unwrap();

    for token in [0, 1] {
        let mut static_state = static_constraint.start();
        let mut dynamic_state = dynamic_constraint.start();
        assert!(static_state.commit_token(token));
        assert!(dynamic_state.commit_token(token));
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

    let spec = ConstraintSpec::builder(parent.clone(), &vocab)
        .unwrap()
        .bind_token("MARK", [7])
        .unwrap()
        .build()
        .unwrap();
    assert_lr_composition_rejected(spec.compile_dynamic());
    let constraint = parent.bind_snapshot("MARK", glrmask::TokenSet::from(7)).unwrap()
        .compile_with_build_options(&vocab, glrmask::test_modes::MaskMode::C.build_options()).unwrap();
    let mut state = constraint.start();
    assert!(state.commit_token(7));
    assert!(state.commit_token(8));
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
    let bound = Grammar::glrm(
        "glrm 1; start start; extern grammar X; nt start = X;",
    ).bind_grammar("X", Grammar::ebnf(r#"start ::= "x""#));
    assert!(bound.is_ok());

    let unknown = Grammar::glrm("glrm 1; start start; nt start = \"x\";")
        .bind_grammar("missing", Grammar::ebnf(r#"start ::= "x""#))
        .unwrap_err()
        .to_string();
    assert!(unknown.contains("no extern named"), "{unknown}");
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
        .compile()
        .unwrap();
    let mut state = source_bound.start();
    assert!(state.commit_token(0));
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

    let unresolved_grammar_child = Grammar::from_glrm(
        "glrm 1; start start; extern grammar nested; nt start = nested;",
    ).unwrap();
    let open = Grammar::from_glrm(parent).unwrap()
        .bind_snapshot("child", unresolved_grammar_child)
        .unwrap()
        .prepare(&vocab)
        .unwrap()
        ;
    // Independent open components may reuse the same private sentinel token
    // number. Terminal IDs, not those hidden token IDs, are the linker
    // coordinate; qualified nested slots must survive serialization and bind.
    let open = PreparedGrammar::load(open.save(), &vocab).unwrap();
    let leaf = Grammar::from_ebnf(r#"start ::= "x""#).unwrap()
        .prepare(&vocab)
        .unwrap();
    let fully_bound = open
        .bind_snapshot("child::nested", &leaf)
        .unwrap()
        .link()
        .unwrap();
    let mut state = fully_bound.start();
    assert!(state.commit_token(0));
    assert!(state.is_accepting());

    let dynamic_fully_bound = open
        .bind_snapshot("child::nested", &leaf)
        .unwrap()
        .link_with(glrmask::test_modes::MaskMode::C.build_options())
        .unwrap();
    let mut state = dynamic_fully_bound.start();
    assert!(state.commit_token(0));
    assert!(state.is_accepting());

    let unresolved_token_child = Grammar::glrm(
        "glrm 1; start start; extern token TOKEN; nt start = TOKEN;",
    );
    let spec = ConstraintSpec::builder(Grammar::glrm(parent), &vocab)
        .unwrap()
        .bind_grammar("child", unresolved_token_child)
        .unwrap()
        .build()
        .unwrap();
    assert!(spec.compile().is_err());
}

#[test]
fn incompatible_compiled_child_target_is_rejected_at_bind_time() {
    let child_vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let parent_vocab = Vocab::new(vec![(0, b"b".to_vec())]);
    let child = Constraint::compile(
        Grammar::ebnf(r#"start ::= "a""#),
        &child_vocab,
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
    edge begin -> middle: "a";
    edge middle -> done: "b";
};
nt start = WORD;
"#;
    let constraint = Constraint::compile(
        Grammar::glrm(source),
        &vocab,
    )
    .unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 0));
    assert!(allowed(&state.mask(), 1));
    assert!(state.commit_token(0));
    assert!(state.is_accepting());
}

#[test]
fn nested_external_tokens_are_rejected() {
    let vocab = Vocab::new(vec![(123, Vec::new())]);
    let source = r#"
glrm 1;
start document;
grammar inner {
    start value;
    extern token END;
    nt value = END;
};
nt document = inner;
"#;
    let err = ConstraintSpec::builder(Grammar::glrm(source), &vocab).unwrap_err().to_string();
    assert!(err.contains("extern declarations are only allowed at the top level"), "{err}");
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
    edge begin -> done: END;
};
"#;
    let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab)
        .unwrap()
        .bind_token("END", [77])
        .unwrap()
        .build()
        .unwrap();
    assert!(allowed(&spec.compile().unwrap().start().mask(), 77));
    assert!(allowed(&spec.compile_dynamic().unwrap().start().mask(), 77));
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
    let plain = Constraint::compile(Grammar::glrm(plain), &vocab).unwrap();
    let hinted = Constraint::compile(Grammar::glrm(hinted), &vocab).unwrap();
    let mut left = plain.start();
    let mut right = hinted.start();
    assert_eq!(left.mask(), right.mask());
    assert!(left.commit_token(1));
    assert!(right.commit_token(1));
    assert_eq!(left.mask(), right.mask());
}

#[test]
fn extern_without_a_kind_is_rejected_at_the_declaration() {
    let source = "glrm 1;\nstart root;\nextern child;\nnt root = \"x\";\n";
    let error = Grammar::from_glrm(source).unwrap_err();
    assert_eq!(
        error.to_string(),
        "grammar error: external declarations require 'extern token NAME' or 'extern grammar NAME' (line 3, column 13)"
    );
    let span = error.source_span().expect("the rejection points at the declaration");
    let declaration = source.find("extern child;").unwrap();
    assert!(span.start >= declaration, "span {span:?}");
    assert!(span.end <= declaration + "extern child;".len(), "span {span:?}");
    assert_eq!(span.line, 3);
}
