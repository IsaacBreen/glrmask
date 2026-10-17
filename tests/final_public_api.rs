use glrmask::{Grammar, PreparedGrammar, Vocab};

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
    let compiled = child.prepare(&v).unwrap();
    for c in [parent.bind_snapshot("child", &child).unwrap().compile(&v).unwrap(),
              parent.prepare(&v).unwrap().bind_snapshot("child", &compiled).unwrap().link().unwrap()] {
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
    let host = Grammar::from_glrm(HOST).prepare(&v).unwrap();
    let a = Grammar::from_ebnf(r#"start ::= "a""#).prepare(&v).unwrap();
    let b = Grammar::from_ebnf(r#"start ::= "b""#).prepare(&v).unwrap();
    for parent in [&host, &PreparedGrammar::load(host.save()).unwrap()] {
        assert!(parent.link().is_err());
        let bound = parent.bind_snapshot("child", &a).unwrap().link().unwrap();
        assert!(allowed(&bound.start().mask(), 4));
        let c = parent.bind_snapshot("child", &b).unwrap().link().unwrap();
        assert!(!allowed(&c.start().mask(), 4));
        assert!(parent.link().is_err());
    }
}

#[test]
fn source_exact_token_identity_is_not_bytes() {
    let v = vocab();
    let c = Grammar::from_glrm(TOKEN_HOST).bind_snapshot("MARK", glrmask::Token::new(7)).unwrap().compile(&v).unwrap();
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
    let m = g.prepare(&v).unwrap();
    assert!(m.link().is_err());
    let expected = g.bind_snapshot("MARK", v.tokens([7, 8]).unwrap()).unwrap().compile(&v).unwrap();
    let actual = m.bind_snapshot("MARK", v.tokens([7, 8]).unwrap()).unwrap().link().unwrap();
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
    let m = Grammar::from_glrm(TOKEN_HOST).prepare(&v).unwrap();
    let loaded = PreparedGrammar::load(m.save()).unwrap();
    assert!(loaded.link().is_err(), "loading must not silently close unresolved token slots");
    let c = loaded.bind_snapshot("MARK", glrmask::Token::new(7)).unwrap().link().unwrap();
    let mut s = c.start();
    s.commit_token(0).unwrap();
    assert!(allowed(&s.mask(), 7));
    assert!(!allowed(&s.mask(), 1));
}

#[test]
fn token_membership_and_compiled_vocabulary_are_validated() {
    let v = vocab();
    let other = Vocab::new(vec![(7, b"different".to_vec())]);
    let parent = Grammar::from_glrm(TOKEN_HOST);
    assert!(parent.bind_snapshot("MARK", glrmask::Token::new(6)).unwrap().compile(&v).is_err());
    // Widening is lossless: a grammar port can contain one exact-ID event.
    // It must not accidentally admit the byte-identical ID 1 instead of ID 7.
    let widened=Grammar::from_glrm(HOST).bind_snapshot("child",glrmask::Token::new(7)).unwrap().compile(&v).unwrap();
    let mut state=widened.start();state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(),7));assert!(!allowed(&state.mask(),1));
    state.commit_token(7).unwrap();state.commit_token(3).unwrap();assert!(state.is_accepting());
    let child = Grammar::from_ebnf(r#"start ::= "a""#).prepare(&v).unwrap();
    assert!(parent.prepare(&v).unwrap().bind_snapshot("MARK", &child).is_err());
    let incompatible = Grammar::from_ebnf(r#"start ::= "different""#).prepare(&other).unwrap();
    assert!(Grammar::from_glrm(HOST).prepare(&v).unwrap().bind_snapshot("child", &incompatible).is_err());
}

#[test]
fn nested_open_token_module_keeps_its_qualified_kind_after_load() {
    let v = vocab();
    // Source and prepared children may both remain open after attachment.
    // The original typed contract survives qualified binding and persistence.
    let inner = Grammar::from_glrm("glrm 1; start start; extern token MARK; nt start = MARK;");
    let outer = Grammar::from_glrm(HOST);
    let open = outer.bind_snapshot("child", &inner).unwrap().prepare(&v).unwrap();
    assert!(open.link().is_err());
    let loaded = PreparedGrammar::load(open.save()).unwrap();
    let grammar_child = Grammar::from_ebnf(r#"start ::= "a""#).prepare(&v).unwrap();
    assert!(loaded.bind_snapshot("child::MARK", &grammar_child).is_err());
    let closed = loaded.bind_snapshot("child::MARK", v.tokens([7, 8]).unwrap()).unwrap().link().unwrap();
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
    let child_module = child_source.prepare(&v).unwrap();
    assert!(child_module.link().is_err());
    let parent = Grammar::from_glrm(HOST);
    let grammar = parent.bind_snapshot("child", &child_source).unwrap();
    assert!(grammar.compile(&v).is_err());
    let fresh = grammar.prepare(&v).unwrap();
    for open in [fresh.clone(), PreparedGrammar::load(fresh.save()).unwrap()] {
        let open = PreparedGrammar::load_with_vocab(open.save(), &v).unwrap();
        assert!(open.link().is_err());
        let closed = open.bind_snapshot("child::MARK", glrmask::Token::new(7)).unwrap().link().unwrap();
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
    let host = grammar.prepare(&v).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "a""#).prepare(&v).unwrap();
    assert!(host.bind_snapshot("MARK", &child).is_err());
    let widened=host.bind_snapshot("child",glrmask::Token::new(7)).unwrap();
    assert_eq!(widened.requirements().unwrap().iter().map(|r|r.name()).collect::<Vec<_>>(),["MARK"]);
    assert!(widened.link().is_err());
    let widened=widened.bind_snapshot("MARK",glrmask::Token::new(8)).unwrap().link().unwrap();
    let mut state=widened.start();
    state.commit_token(8).unwrap();assert!(allowed(&state.mask(),7));assert!(!allowed(&state.mask(),1));
    state.commit_token(7).unwrap();assert!(state.is_accepting());
    let token_first = host.bind_snapshot("MARK", glrmask::Token::new(8)).unwrap();
    let token_first = PreparedGrammar::load(token_first.save()).unwrap();
    let token_first = token_first.bind_snapshot("child", &child).unwrap().link().unwrap();
    let child_first = host.bind_snapshot("child", &child).unwrap();
    let child_first = PreparedGrammar::load(child_first.save()).unwrap();
    let child_first = child_first.bind_snapshot("MARK", glrmask::Token::new(8)).unwrap().link().unwrap();
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
    let module = Grammar::from_glrm(TOKEN_HOST).prepare(&v).unwrap();
    let bytes = module.save();
    assert!(Constraint::load(bytes.clone()).is_err());
    assert!(bytes.starts_with(b"GLRMOD"));
    // An open package is never runnable, even when its root body has byte paths.
    assert!(Constraint::load(bytes.clone()).is_err());
    let constraint = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    assert!(PreparedGrammar::load(constraint.save()).is_err());
    for length in [0, 1, 8, 15, 16, bytes.len() - 1] {
        assert!(PreparedGrammar::load(&bytes[..length]).is_err(), "accepted truncation to {length}");
    }
    let mut corrupt = bytes.clone();
    corrupt[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(PreparedGrammar::load(corrupt).is_err());
    let incompatible = Vocab::new(vec![(7, b"different".to_vec())]);
    assert!(PreparedGrammar::load_with_vocab(bytes, &incompatible).is_err());
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
        let expected = grammar.bind_snapshot("MARK", v.tokens([7, 8]).unwrap()).unwrap().compile(&v).unwrap();
        let module = grammar.prepare(&v).unwrap();
        let module = PreparedGrammar::load(module.save()).unwrap();
        let bound = module.bind_snapshot("MARK", v.tokens([7, 8]).unwrap()).unwrap();
        let unsaved = bound.link().unwrap();
        assert_same_language(&expected, &unsaved, 4);
        let loaded_module = PreparedGrammar::load(bound.save()).unwrap();
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
    let child_open = child.prepare(&v).unwrap();
    let parent = Grammar::from_glrm(
        "glrm 1; start start; extern grammar A; extern grammar B; nt start = A B;",
    );
    let open = parent.bind_snapshot("A", &child).unwrap().bind_snapshot("B", &child).unwrap()
        .prepare(&v).unwrap();
    let loaded = PreparedGrammar::load(open.save()).unwrap();
    let first = loaded.bind_snapshot("A::MARK", glrmask::Token::new(7)).unwrap();
    assert!(first.link().is_err());
    let final_module = first.bind_snapshot("B::MARK", glrmask::Token::new(8)).unwrap();
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
    let module = Grammar::from_ebnf(r#"start ::= "a""#).prepare(&v).unwrap();
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
        assert!(state.is_accepting() && !state.is_rejected());
        assert_eq!(state.mask(), vec![0; 8]);
        assert!(state.forced_canonical_reference().is_empty());
        assert!(state.commit_token(1).is_err());
        assert!(state.commit_bytes(b"a").is_err());
        assert!(other.is_accepting() && !other.is_rejected());
        assert!(allowed(&other.mask(), 130));
        assert!(allowed(&other.mask(), 255));
        other.commit_token(255).unwrap();
        assert!(other.is_accepting() && !other.is_rejected());
        assert_eq!(other.mask(), vec![0; 8]);
        assert!(other.commit_token(1).is_err());
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
    assert!(!state.is_accepting());
    assert!(state.mask().iter().all(|word| *word == 0));
    let mut valid = constraint.start();
    valid.commit_token(1).unwrap();
    assert!(allowed(&valid.mask(), 7));
    valid.commit_token(7).unwrap();
    assert!(valid.is_accepting() && !valid.is_rejected());
    assert!(valid.mask().iter().all(|word| *word == 0));
    assert!(valid.commit_token(1).is_err());
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
        assert!(state.is_accepting() && !state.is_rejected());
        assert!(state.commit_token(1).is_err());
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
    assert!(PreparedGrammar::load(fake_module).is_err());
}

#[test]
fn public_constraint_load_rejects_an_open_module_body() {
    let v = vocab();
    let module = Grammar::from_glrm(
        r#"glrm 1; start start; extern grammar CHILD; nt start = CHILD;"#,
    )
    .prepare(&v)
    .unwrap();
    let bytes = module.save();
    assert!(bytes.starts_with(b"GLRMOD"));
    assert!(glrmask::Constraint::load(bytes.clone()).is_err());
    assert!(PreparedGrammar::load(bytes).is_ok());
}

#[test]
fn fast_build_module_link_is_safe_in_a_single_worker_rayon_pool() {
    use glrmask::{BuildOptions, Optimization};

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    // Build and consume the whole runtime inside the pool. The public API does
    // not promise that PreparedGrammar/Constraint are Send; this regression is about
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
        .prepare(&v)
        .unwrap();
        let child = Grammar::from_ebnf(r#"start ::= "a""#)
            .prepare(&v)
            .unwrap();
        let bound = host.bind_snapshot("CHILD", &child).unwrap();
        let constraint = bound
            .link_with(
                BuildOptions::default().optimization(Optimization::Balanced),
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
        .prepare(&v)
        .unwrap()
        .bind_snapshot("MARK", v.tokens([7, 8]).unwrap())
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
    .prepare(&v)
    .unwrap()
    .bind_snapshot("MARK", glrmask::Token::new(7))
    .unwrap();
    let parent = Grammar::from_glrm(
        r#"glrm 1; start start; extern grammar child; nt start = "x" child "y";"#,
    )
    .prepare(&v).unwrap()
    .bind_snapshot("child", &token_child)
    .unwrap();
    let expected = Grammar::from_ebnf(r#"start ::= "x" @token(7) "y""#)
        .compile(&v)
        .unwrap();

    for optimization in [Optimization::Balanced, Optimization::Auto, Optimization::FastRuntime] {
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

    let open = source.prepare(&vocab).unwrap();
    let loaded_open = PreparedGrammar::load(open.save()).unwrap();
    let bound = loaded_open
        .bind_snapshot("CONTROL", glrmask::Token::new(2))
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
    assert!(PreparedGrammar::load_with_vocab(open.save(), &missing_domain).is_err());
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
    .bind_snapshot("CONTROL", glrmask::Token::new(77))
    .unwrap()
    .compile(&vocab)
    .unwrap();
    assert!(allowed(&special.start().mask(), 77));
}

#[test]
fn typed_ports_and_explicit_event_adapters_preserve_semantics_in_every_mode() {
    use glrmask::{BuildOptions, Optimization, Token};
    let v = Vocab::new_with_exact_token_ids(
        vec![(0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"y".to_vec()),
             (3, b"xay".to_vec()), (7, b"a".to_vec())], [63]);
    let source = Grammar::from_glrm(
        r#"glrm 1; start s; extern token item; nt s = "x" item "y";"#);
    let fragments = Grammar::from_glrm(r#"extern grammar item start s nt s = "x" item "y""#);
    for mode in [Optimization::Auto, Optimization::FastBuild,
                 Optimization::Balanced, Optimization::FastRuntime] {
        let options = BuildOptions::default().optimization(mode);
        for id in [7, 63] {
            let exact = source.bind_snapshot("item", Token::new(id)).unwrap()
                .compile_with(&v, options.clone()).unwrap();
            let linked = PreparedGrammar::load(source.prepare(&v).unwrap().save()).unwrap()
                .bind_snapshot("item", Token::new(id)).unwrap().link_with(options.clone()).unwrap();
            let mut a = exact.start(); let mut b = linked.start();
            assert!(!allowed(&a.mask(), 3));
            for token in [0, id, 2] {
                assert_eq!(a.mask(), b.mask());
                assert!(allowed(&a.mask(), token));
                if token == id { assert!(!allowed(&a.mask(), 1)); }
                a.commit_token(token).unwrap(); b.commit_token(token).unwrap();
            }
            assert!(a.is_accepting() && b.is_accepting());
            let adapter = Grammar::exact(v.tokens([id]).unwrap());
            let adapted = fragments.bind_snapshot("item", &adapter).unwrap().compile_with(&v, options.clone()).unwrap();
            let mut c = adapted.start();
            assert!(!allowed(&c.mask(), 3));
            for token in [0,id,2] { assert!(allowed(&c.mask(),token)); c.commit_token(token).unwrap(); }
            assert!(c.is_accepting());
        }
        let child = Grammar::from_ebnf(r#"s ::= "a""#);
        let text = fragments.bind_snapshot("item", &child).unwrap().compile_with(&v, options.clone()).unwrap();
        assert!(allowed(&text.start().mask(), 3));
        let module = fragments.prepare(&v).unwrap();
        let compiled = child.prepare(&v).unwrap();
        let text = module.bind_snapshot("item", &compiled).unwrap().link_with(options).unwrap();
        assert!(allowed(&text.start().mask(), 3));
    }
}

#[test]
fn token_constructor_defers_sparse_domain_validation() {
    use glrmask::Token;
    let v = Vocab::new_with_exact_token_ids(vec![(2, b"a".to_vec())], [77]);
    let source = Grammar::from_glrm("glrm 1; start s; extern token item; nt s = item;");
    let module = source.prepare(&v).unwrap();
    for id in [0, 3, 76, 78, u32::MAX] {
        let value = Token::new(id);
        assert_eq!(value.id(), id);
        let bound = source.bind_snapshot("item", value).unwrap();
        assert!(bound.compile(&v).unwrap_err().to_string().contains("not present"));
        assert!(bound.prepare(&v).is_err());
        assert!(module.bind_snapshot("item", value).unwrap_err().to_string().contains("not present"));
    }
    assert!(source.bind_snapshot("item", Token::new(77)).unwrap().compile(&v).is_ok());
}
