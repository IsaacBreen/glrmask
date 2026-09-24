use glrmask::{Grammar, UnlinkedConstraint, Vocab};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
        (3, b"y".to_vec()), (4, b"xay".to_vec()), (5, b"ab".to_vec()),
        (7, b"a".to_vec()), (8, Vec::new()),
    ])
}
fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|w| w & (1 << (id % 32)) != 0)
}
const HOST: &str = "glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";";
const TOKEN_HOST: &str = "glrm 1; start start; extern token MARK; nt start = \"x\" MARK \"y\";";

#[test]
fn immutable_source_and_compiled_bindings_cross_token_boundaries() {
    let v = vocab();
    let parent = Grammar::from_glrm(HOST);
    let child = Grammar::from_ebnf(r#"start ::= "a""#);
    let compiled = child.compile(&v).unwrap();
    for c in [parent.bind("child", &child).unwrap().compile(&v).unwrap(),
              parent.compile_unlinked(&v).unwrap().bind("child", &compiled).unwrap().link().unwrap()] {
        let mut s = c.start();
        assert!(allowed(&s.mask(), 4));
        s.commit_token(4).unwrap();
        assert!(s.is_accepting());
    }
    assert!(parent.compile(&v).is_err());
}

#[test]
fn module_reuse_load_and_closed_link() {
    let v = vocab();
    let host = Grammar::from_glrm(HOST).compile_unlinked(&v).unwrap();
    let a = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    let b = Grammar::from_ebnf(r#"start ::= "b""#).compile_unlinked(&v).unwrap().link().unwrap();
    for parent in [&host, &UnlinkedConstraint::load(host.save()).unwrap()] {
        assert!(parent.link().is_err());
        let bound = parent.bind("child", &a).unwrap().link().unwrap();
        assert!(allowed(&bound.start().mask(), 4));
        let c = parent.bind("child", &b).unwrap().link().unwrap();
        assert!(!allowed(&c.start().mask(), 4));
        assert!(parent.link().is_err());
    }
}

#[test]
fn source_exact_token_identity_is_not_bytes() {
    let v = vocab();
    let c = Grammar::from_glrm(TOKEN_HOST).bind("MARK", v.token(7).unwrap()).unwrap().compile(&v).unwrap();
    let mut s = c.start();
    assert!(!allowed(&s.mask(), 4));
    s.commit_token(0).unwrap();
    assert!(allowed(&s.mask(), 7));
    assert!(!allowed(&s.mask(), 1));
    s.commit_token(7).unwrap();
    s.commit_token(3).unwrap();
    assert!(s.is_accepting());
}

#[test]
fn compiled_exact_token_bind_matches_source_at_every_step() {
    let v = vocab();
    let g = Grammar::from_glrm(TOKEN_HOST);
    let m = g.compile_unlinked(&v).unwrap();
    assert!(m.link().is_err());
    let expected = g.bind("MARK", v.tokens([7, 8]).unwrap()).unwrap().compile(&v).unwrap();
    let actual = m.bind("MARK", v.tokens([7, 8]).unwrap()).unwrap().link().unwrap();
    let mut a = actual.start(); let mut b = expected.start();
    for id in [0, 7, 3] {
        assert_eq!(a.mask(), b.mask(), "before {id}");
        a.commit_token(id).unwrap(); b.commit_token(id).unwrap();
        assert_eq!(a.is_accepting(), b.is_accepting());
    }
    assert_eq!(a.mask(), b.mask());
    assert!(m.link().is_err());
}

#[test]
fn unbound_exact_token_survives_save_load() {
    let v = vocab();
    let m = Grammar::from_glrm(TOKEN_HOST).compile_unlinked(&v).unwrap();
    let loaded = UnlinkedConstraint::load(m.save()).unwrap();
    assert!(loaded.link().is_err(), "loading must not silently close unresolved token slots");
    let c = loaded.bind("MARK", v.token(7).unwrap()).unwrap().link().unwrap();
    let mut s = c.start();
    s.commit_token(0).unwrap();
    assert!(allowed(&s.mask(), 7));
    assert!(!allowed(&s.mask(), 1));
}

#[test]
fn wrong_vocabulary_and_slot_kind_are_rejected() {
    let v = vocab();
    let other = Vocab::new(vec![(7, b"different".to_vec())]);
    let parent = Grammar::from_glrm(TOKEN_HOST);
    assert!(parent.bind("MARK", other.token(7).unwrap()).unwrap().compile(&v).is_err());
    assert!(Grammar::from_glrm(HOST).bind("child", v.token(7).unwrap()).is_err());
    let child = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    assert!(parent.compile_unlinked(&v).unwrap().bind("MARK", &child).is_err());
    let incompatible = Grammar::from_ebnf(r#"start ::= "different""#).compile(&other).unwrap();
    assert!(Grammar::from_glrm(HOST).compile_unlinked(&v).unwrap().bind("child", &incompatible).is_err());
}

#[test]
fn nested_open_token_module_keeps_its_qualified_kind_after_load() {
    let v = vocab();
    // Open child graphs are assembled in the source world; the public compiled
    // binding API now accepts runnable children, not open child artifacts.
    let inner = Grammar::from_glrm("glrm 1; start start; extern token MARK; nt start = MARK;");
    let outer = Grammar::from_glrm(HOST);
    let open = outer.bind("child", &inner).unwrap().compile_unlinked(&v).unwrap();
    assert!(open.link().is_err());
    let loaded = UnlinkedConstraint::load(open.save()).unwrap();
    let grammar_child = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    assert!(loaded.bind("child.MARK", &grammar_child).is_err());
    let closed = loaded.bind("child.MARK", v.tokens([7, 8]).unwrap()).unwrap().link().unwrap();
    for token in [7, 8] {
        let mut state = closed.start();
        for id in [0, token, 3] {
            assert!(allowed(&state.mask(), id), "missing {id} in nested token path");
            state.commit_token(id).unwrap();
        }
        assert!(state.is_accepting());
    }
    assert!(loaded.link().is_err());
}

#[test]
fn mixed_source_compilation_never_silently_closes_child_token_slots() {
    let v = vocab();
    let child_source = Grammar::from_glrm("glrm 1; start start; extern token MARK; nt start = MARK;");
    let child_module = child_source.compile_unlinked(&v).unwrap();
    assert!(child_module.link().is_err());
    let parent = Grammar::from_glrm(HOST);
    let grammar = parent.bind("child", &child_source).unwrap();
    assert!(grammar.compile(&v).is_err());
    let fresh = grammar.compile_unlinked(&v).unwrap();
    for open in [fresh.clone(), UnlinkedConstraint::load(fresh.save()).unwrap()] {
        let open = UnlinkedConstraint::load_with_vocab(open.save(), &v).unwrap();
        assert!(open.link().is_err());
        let closed = open.bind("child.MARK", v.token(7).unwrap()).unwrap().link().unwrap();
        let mut s = closed.start();
        s.commit_token(0).unwrap();
        assert!(allowed(&s.mask(), 7));
        assert!(!allowed(&s.mask(), 1));
    }
}

#[test]
fn mixed_slot_binding_order_and_saved_partial_modules() {
    let v = vocab();
    let grammar = Grammar::from_glrm(
        "glrm 1; start start; extern token MARK; extern grammar child; nt start = MARK child;",
    );
    let host = grammar.compile_unlinked(&v).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    assert!(host.bind("MARK", &child).is_err());
    assert!(host.bind("child", v.token(7).unwrap()).is_err());
    let token_first = host.bind("MARK", v.token(8).unwrap()).unwrap();
    let token_first = UnlinkedConstraint::load(token_first.save()).unwrap();
    let token_first = token_first.bind("child", &child).unwrap().link().unwrap();
    let child_first = host.bind("child", &child).unwrap();
    let child_first = UnlinkedConstraint::load(child_first.save()).unwrap();
    let child_first = child_first.bind("MARK", v.token(8).unwrap()).unwrap().link().unwrap();
    let mut a = token_first.start();
    let mut b = child_first.start();
    for id in [8, 1] {
        assert_eq!(a.mask(), b.mask());
        assert!(allowed(&a.mask(), id));
        a.commit_token(id).unwrap();
        b.commit_token(id).unwrap();
    }
    assert!(a.is_accepting() && b.is_accepting());
    assert!(host.link().is_err());
}

#[test]
fn artifact_kinds_and_corrupt_module_headers_are_rejected() {
    use glrmask::Constraint;
    let v = vocab();
    let module = Grammar::from_glrm(TOKEN_HOST).compile_unlinked(&v).unwrap();
    let bytes = module.save();
    assert!(Constraint::load(bytes.clone()).is_err());
    let manifest_len =
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let body_start = 16 + manifest_len;
    assert!(
        Constraint::load(&bytes[body_start..]).is_err(),
        "an open UnlinkedConstraint body must not be loadable as a public Constraint",
    );
    let constraint = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    assert!(UnlinkedConstraint::load(constraint.save()).is_err());
    for length in [0, 1, 8, 15, 16, bytes.len() - 1] {
        assert!(UnlinkedConstraint::load(&bytes[..length]).is_err(), "accepted truncation to {length}");
    }
    let mut corrupt = bytes.clone();
    corrupt[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(UnlinkedConstraint::load(corrupt).is_err());
    let incompatible = Vocab::new(vec![(7, b"different".to_vec())]);
    assert!(UnlinkedConstraint::load_with_vocab(bytes, &incompatible).is_err());
}

fn assert_same_language(expected: &glrmask::Constraint, actual: &glrmask::Constraint, depth: usize) {
    let mut frontier = vec![(expected.start(), actual.start(), Vec::<u32>::new())];
    for step in 0..=depth {
        let mut next = Vec::new();
        for (a, b, prefix) in frontier {
            assert_eq!(a.mask(), b.mask(), "mask after {prefix:?}");
            assert_eq!(a.is_accepting(), b.is_accepting(), "acceptance after {prefix:?}");
            assert_eq!(a.is_rejected(), b.is_rejected(), "rejection after {prefix:?}");
            if step == depth { continue; }
            for id in 0..9 {
                if !allowed(&a.mask(), id) { continue; }
                let mut aa = a.clone(); let mut bb = b.clone();
                aa.commit_token(id).unwrap(); bb.commit_token(id).unwrap();
                let mut path = prefix.clone(); path.push(id);
                next.push((aa, bb, path));
            }
        }
        frontier = next;
    }
}

#[test]
fn exact_token_link_is_differentially_equal_to_source_for_choices_and_repetition() {
    let v = vocab();
    for rule in ["MARK", "MARK MARK", "MARK | eps", "MARK \"b\" | \"a\" \"y\"", "\"x\" MARK \"y\""] {
        let source = format!("glrm 1; start start; extern token MARK; nt start = {rule};");
        let grammar = Grammar::from_glrm(&source);
        let expected = grammar.bind("MARK", v.tokens([7, 8]).unwrap()).unwrap().compile(&v).unwrap();
        let module = grammar.compile_unlinked(&v).unwrap();
        let module = UnlinkedConstraint::load(module.save()).unwrap();
        let bound = module.bind("MARK", v.tokens([7, 8]).unwrap()).unwrap();
        let unsaved = bound.link().unwrap();
        assert_same_language(&expected, &unsaved, 4);
        let loaded_module = UnlinkedConstraint::load(bound.save()).unwrap();
        let actual = loaded_module.link().unwrap();
        let loaded = glrmask::Constraint::load(actual.save()).unwrap();
        assert_same_language(&expected, &actual, 4);
        assert_same_language(&expected, &loaded, 4);
    }
}

#[test]
fn repeated_open_child_slots_keep_independent_token_bindings() {
    let v = vocab();
    let child = Grammar::from_glrm("glrm 1; start start; extern token MARK; nt start = MARK;");
    let child_open = child.compile_unlinked(&v).unwrap();
    let parent = Grammar::from_glrm(
        "glrm 1; start start; extern grammar A; extern grammar B; nt start = A B;",
    );
    let open = parent.bind("A", &child).unwrap().bind("B", &child).unwrap()
        .compile_unlinked(&v).unwrap();
    let loaded = UnlinkedConstraint::load(open.save()).unwrap();
    let first = loaded.bind("A.MARK", v.token(7).unwrap()).unwrap();
    assert!(first.link().is_err());
    let final_module = first.bind("B.MARK", v.token(8).unwrap()).unwrap();
    let constraint = final_module.link().unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 7));
    assert!(!allowed(&state.mask(), 8));
    state.commit_token(7).unwrap();
    assert!(allowed(&state.mask(), 8));
    assert!(!allowed(&state.mask(), 7));
    state.commit_token(8).unwrap();
    assert!(state.is_accepting());
    assert!(child_open.link().is_err());
}

#[test]
fn root_end_tokens_can_extend_the_mask_without_changing_body_caches() {
    use glrmask::BuildOptions;
    let v = vocab();
    let module = Grammar::from_ebnf(r#"start ::= "a""#).compile_unlinked(&v).unwrap();
    let mut constraint = module.link_with(BuildOptions::default().end_tokens([130, 255])).unwrap();
    for _ in 0..4 {
        assert_eq!(constraint.mask_len(), 8);
        let mut state = constraint.start();
        let mut buffer = vec![u32::MAX; 10];
        state.fill_mask(&mut buffer);
        assert_eq!(&buffer[..8], state.mask());
        assert_eq!(&buffer[8..], &[0, 0]);
        assert!(!allowed(&state.mask(), 130));
        state.commit_token(1).unwrap();
        assert!(state.is_accepting());
        assert!(allowed(&state.mask(), 130));
        assert!(allowed(&state.mask(), 255));
        let mut other = state.clone();
        state.commit_token(130).unwrap();
        assert!(state.is_terminated() && state.is_accepting() && !state.is_rejected());
        assert_eq!(state.mask(), vec![0; 8]);
        assert!(state.forced().is_empty());
        assert!(state.commit_token(1).is_err());
        assert!(state.commit_bytes(b"a").is_err());
        assert!(!other.is_terminated());
        other.commit_token(255).unwrap();
        assert!(other.is_terminated());
        constraint = glrmask::Constraint::load_with_vocab(constraint.save(), &v).unwrap();
    }
    assert_eq!(module.link().unwrap().mask_len(), 1);
    let policy_b = module.link_with(BuildOptions::default().end_tokens([63])).unwrap();
    assert_eq!(policy_b.mask_len(), 2);
}

#[test]
fn root_end_ids_are_reserved_and_early_end_rejects_even_with_matching_bytes() {
    use glrmask::BuildOptions;
    let v = vocab();
    // IDs 1 and 7 share bytes, but only 7 is reserved as generation control.
    let constraint = Grammar::from_ebnf(r#"start ::= "a""#)
        .compile_with(&v, BuildOptions::default().end_tokens([7])).unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 1));
    assert!(!allowed(&state.mask(), 7));
    state.commit_token(7).unwrap();
    assert!(state.is_rejected());
    assert!(!state.is_accepting() && !state.is_terminated());
    assert!(state.mask().iter().all(|word| *word == 0));
    let mut valid = constraint.start();
    valid.commit_token(1).unwrap();
    assert!(allowed(&valid.mask(), 7));
    valid.commit_token(7).unwrap();
    assert!(valid.is_terminated());
    assert!(Grammar::from_ebnf(r#"start ::= "a""#)
        .compile_with(&v, BuildOptions::default().end_tokens([7, 7])).is_err());
}

#[test]
fn root_policy_mask_and_commit_profilers_preserve_termination() {
    use glrmask::BuildOptions;
    use glrmask::__private::ConstraintStateExt;
    let v = vocab();
    let constraint = Grammar::from_ebnf(r#"start ::= "a""#)
        .compile_with(&v, BuildOptions::default().end_tokens([130])).unwrap();
    for mode in 0..3 {
        let mut state = constraint.start();
        state.commit_token(1).unwrap();
        let mut buffer = vec![u32::MAX; constraint.mask_len() + 1];
        state.fill_mask_profiled(&mut buffer);
        assert!(allowed(&buffer, 130));
        assert_eq!(buffer.last(), Some(&0));
        match mode {
            0 => { state.commit_token_timed_ns(130).unwrap(); }
            1 => { state.commit_token_profiled(130).unwrap(); }
            _ => { state.commit_token_per_advance(130).unwrap(); }
        }
        assert!(state.is_terminated());
        state.fill_mask_profiled(&mut buffer);
        assert!(buffer.iter().all(|word| *word == 0));
    }
}

#[test]
fn root_artifact_headers_are_bounded_and_cannot_enter_module_body() {
    use glrmask::BuildOptions;
    let v = vocab();
    let constraint = Grammar::from_ebnf(r#"start ::= "a""#)
        .compile_with(&v, BuildOptions::default().end_tokens([127])).unwrap();
    let bytes = constraint.save();
    for len in [0, 1, 7, 8, 11, 12, 15, 16, bytes.len() - 1] {
        assert!(glrmask::Constraint::load(&bytes[..len]).is_err());
    }
    let mut oversized = bytes.clone();
    oversized[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(glrmask::Constraint::load(oversized).is_err());
    let mut fake_module = b"GLRMOD01".to_vec();
    fake_module.extend_from_slice(&8u64.to_le_bytes());
    fake_module.extend_from_slice(&0u64.to_le_bytes());
    fake_module.extend_from_slice(&bytes);
    assert!(UnlinkedConstraint::load(fake_module).is_err());
}

#[test]
fn public_constraint_load_rejects_an_open_module_body() {
    let v = vocab();
    let module = Grammar::from_glrm(
        r#"glrm 1; start start; extern grammar CHILD; nt start = CHILD;"#,
    )
    .compile_unlinked(&v)
    .unwrap();
    let bytes = module.save();
    assert!(bytes.starts_with(b"GLRMOD03"));
    let manifest_len =
        u64::from_le_bytes(bytes[8..16].try_into().expect("module header")) as usize;
    let body_start = 16 + manifest_len;
    assert!(body_start < bytes.len());
    assert!(glrmask::Constraint::load(&bytes[body_start..]).is_err());
    assert!(UnlinkedConstraint::load(bytes).is_ok());
}

#[test]
fn fast_build_module_link_is_safe_in_a_single_worker_rayon_pool() {
    use glrmask::{BuildOptions, Optimization};

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    // Build and consume the whole runtime inside the pool. The public API does
    // not promise that UnlinkedConstraint/Constraint are Send; this regression is about
    // scheduler progress with one Rayon worker, not cross-thread transport.
    pool.install(|| {
        let v = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"a".to_vec()),
            (2, b"y".to_vec()),
            (3, b"xay".to_vec()),
        ]);
        let host = Grammar::from_glrm(
            r#"glrm 1; start start; extern grammar CHILD; nt start = "x" CHILD "y";"#,
        )
        .compile_unlinked(&v)
        .unwrap();
        let child = Grammar::from_ebnf(r#"start ::= "a""#)
            .compile(&v)
            .unwrap();
        let bound = host.bind("CHILD", &child).unwrap();
        let constraint = bound
            .link_with(
                BuildOptions::default().optimization(Optimization::FastBuild),
            )
            .unwrap();
        let mut state = constraint.start();
        assert!(allowed(&state.mask(), 3));
        state.commit_token(3).unwrap();
        assert!(state.is_accepting());
    });
}

#[test]
fn composed_dynamic_reference_does_not_admit_empty_byte_alias_without_special_path() {
    use glrmask::__private::ConstraintStateExt;

    let v = vocab();
    let source = Grammar::from_glrm(
        r#"glrm 1; start start; extern token MARK; nt start = MARK "b" | "a" "y";"#,
    );
    let constraint = source
        .compile_unlinked(&v)
        .unwrap()
        .bind("MARK", v.tokens([7, 8]).unwrap())
        .unwrap()
        .link()
        .unwrap();

    let mut state = constraint.start();
    assert_eq!(state.mask(), state.fill_mask_dynamic_vec());
    state.commit_token(1).unwrap();
    // Token 8 has empty vocabulary bytes. Its exact MARK path is parser-dead
    // here, so a radix-root endpoint must not make the byte path admit it.
    assert!(!allowed(&state.mask(), 8));
    assert_eq!(state.mask(), state.fill_mask_dynamic_vec());
}

#[test]
fn one_shot_optimization_preserves_deferred_compiled_module_bindings() {
    use glrmask::{BuildOptions, Optimization};

    let v = Vocab::new(vec![
        (0, b"x".to_vec()),
        (1, b"a".to_vec()),
        (2, b"b".to_vec()),
        (3, b"y".to_vec()),
        (4, b"xay".to_vec()),
        (7, b"a".to_vec()),
    ]);
    let token_child = Grammar::from_glrm(
        r#"glrm 1; start start; extern token MARK; nt start = MARK;"#,
    )
    .compile_unlinked(&v)
    .unwrap()
    .bind("MARK", v.token(7).unwrap())
    .unwrap().link().unwrap();
    let parent = Grammar::from_glrm(
        r#"glrm 1; start start; extern grammar child; nt start = "x" child "y";"#,
    )
    .compile_unlinked(&v).unwrap()
    .bind("child", &token_child)
    .unwrap();
    let expected = Grammar::from_ebnf(r#"start ::= "x" @token(7) "y""#)
        .compile(&v)
        .unwrap();

    for optimization in [Optimization::FastBuild, Optimization::Auto, Optimization::FastRuntime] {
        let actual = parent
            .link_with(
                BuildOptions::default().optimization(optimization),
            )
            .unwrap();
        let mut actual_state = actual.start();
        let mut expected_state = expected.start();
        for token in [0, 7, 3] {
            assert_eq!(actual_state.mask(), expected_state.mask());
            actual_state.commit_token(token).unwrap();
            expected_state.commit_token(token).unwrap();
        }
        assert_eq!(actual_state.is_accepting(), expected_state.is_accepting());
    }
}

#[test]
fn exact_only_model_tokens_survive_open_module_and_constraint_roundtrips() {
    let vocab = Vocab::new_with_exact_token_ids(
        vec![(0, b"x".to_vec()), (1, b"y".to_vec())],
        [2, 77],
    );
    let source = Grammar::from_glrm(
        r#"glrm 1; start start; extern token CONTROL; nt start = "x" CONTROL "y";"#,
    );

    let open = source.compile_unlinked(&vocab).unwrap();
    let loaded_open = UnlinkedConstraint::load(open.save()).unwrap();
    let bound = loaded_open
        .bind("CONTROL", vocab.token(2).unwrap())
        .unwrap();
    let constraint = bound.link().unwrap();

    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 2));
    assert!(!allowed(&state.mask(), 77));
    state.commit_token(2).unwrap();
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());

    let loaded = glrmask::Constraint::load(constraint.save()).unwrap();
    let mut state = loaded.start();
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 2));
    state.commit_token(2).unwrap();
    state.commit_token(1).unwrap();
    assert!(state.is_accepting());

    let missing_domain = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
    assert!(UnlinkedConstraint::load_with_vocab(open.save(), &missing_domain).is_err());
    assert!(glrmask::Constraint::load_with_vocab(constraint.save(), &missing_domain).is_err());
}

#[test]
fn exact_only_ids_never_enter_the_byte_language() {
    let vocab = Vocab::new_with_exact_token_ids(
        vec![(0, b"x".to_vec()), (1, b"y".to_vec())],
        [77],
    );
    let plain = Grammar::from_ebnf(r#"start ::= "x" "y""#)
        .compile(&vocab)
        .unwrap();
    assert!(!allowed(&plain.start().mask(), 77));

    let special = Grammar::from_glrm(
        r#"glrm 1; start start; extern token CONTROL; nt start = CONTROL;"#,
    )
    .bind("CONTROL", vocab.token(77).unwrap())
    .unwrap()
    .compile(&vocab)
    .unwrap();
    assert!(allowed(&special.start().mask(), 77));
}
