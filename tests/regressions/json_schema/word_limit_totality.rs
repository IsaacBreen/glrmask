//! A slice certificate must cover every input atom, not only surviving edges.
//! These public-API regressions run with ordinary defaults (no environment setup).
use glrmask::{BuildOptions, Constraint, Grammar, UnlinkedConstraint, Optimization, Vocab};

fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32)
        .is_some_and(|word| word & (1 << (id % 32)) != 0)
}

fn vocabulary() -> Vocab {
    Vocab::new(vec![
        (0, b"\"".to_vec()),
        (1, b"a".to_vec()),
        (2, b" a".to_vec()),
        (3, b"aaa".to_vec()),
        (4, b" because".to_vec()),
        (5, b" ".to_vec()),
        (6, "é".as_bytes().to_vec()),
        // Canonical escaped quote, not an arbitrary Unicode escape alias:
        // the default JSON lexical mode intentionally rejects e.g. \u0061.
        (7, b"\\\"".to_vec()),
        (8, b"[".to_vec()),
        (9, b"]".to_vec()),
        (10, b"\"a a".to_vec()),
    ])
}

fn schema(words: usize) -> String {
    serde_json::json!({
        "type": "string",
        "maxLength": 64,
        "pattern": format!(r"^(?:\S+\s+){{0,{}}}\S+$", words - 1),
    }).to_string()
}

fn assert_last_word_masks(constraint: &Constraint, words: usize) {
    let mut state = constraint.start();
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    for _ in 1..words {
        assert!(allowed(&state.mask(), 2), "the next word still fits");
        state.commit_token(2).unwrap();
    }
    let mask = state.mask();
    for id in [2, 4, 5] {
        assert!(!allowed(&mask, id), "must not offer another separator: {id}");
    }
    // Continuing the final word, including a Unicode scalar or JSON escape,
    // remains legal. Rejecting the whole residual would not be a valid fix.
    for id in [1, 3, 6, 7] {
        assert!(allowed(&mask, id), "legal final-word continuation missing: {id}");
        let mut next = state.clone();
        next.commit_token(id).unwrap();
        assert!(allowed(&next.mask(), 0));
        next.commit_token(0).unwrap();
        assert!(next.is_accepting());
    }
    assert!(allowed(&mask, 0));
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn word_limit_totality_default_and_loaded_public_constraints() {
    let vocab = vocabulary();
    for words in 1..=4 {
        let source = schema(words);
        for mode in [Optimization::Auto, Optimization::FastBuild, Optimization::FastRuntime] {
            let constraint = Grammar::from_json_schema(&source)
                .compile_with(&vocab, BuildOptions::default().optimization(mode))
                .unwrap();
            assert_last_word_masks(&constraint, words);
            let loaded = Constraint::load(constraint.save()).unwrap();
            assert_last_word_masks(&loaded, words);
        }
    }
}

#[test]
fn word_limit_totality_survives_compiled_child_binding_and_module_load() {
    let vocab = vocabulary();
    let source = schema(2);
    let child = Grammar::from_json_schema(&source)
        .compile_with(&vocab, BuildOptions::default().optimization(Optimization::FastBuild))
        .unwrap();
    let host = Grammar::from_glrm(
        r#"glrm 1; start start; extern grammar child; nt start = "[" child "]";"#,
    ).compile_unlinked(&vocab).unwrap();
    let loaded_host = UnlinkedConstraint::load(host.save()).unwrap();
    for parent in [&host, &loaded_host] {
        let bound = parent.bind("child", &child).unwrap().link().unwrap();
        let mut state = bound.start();
        for id in [8, 10] {
            assert!(allowed(&state.mask(), id));
            state.commit_token(id).unwrap();
        }
        for id in [2, 4, 5] {
            assert!(!allowed(&state.mask(), id), "bound child offered separator {id}");
        }
        for id in [0, 9] {
            assert!(allowed(&state.mask(), id));
            state.commit_token(id).unwrap();
        }
        assert!(state.is_accepting());

        // Repeat the support check through an already-linked, serialized
        // component. The virtual lexer is now a nested leaf, not the root.
        let saved_bound = Constraint::load(bound.save()).unwrap();
        let nested = parent.bind("child", &saved_bound).unwrap().link().unwrap();
        let mut state = nested.start();
        for id in [8, 8, 10] {
            assert!(allowed(&state.mask(), id));
            state.commit_token(id).unwrap();
        }
        for id in [2, 4, 5] {
            assert!(!allowed(&state.mask(), id), "nested child offered separator {id}");
        }
        for id in [0, 9, 9] {
            assert!(allowed(&state.mask(), id));
            state.commit_token(id).unwrap();
        }
        assert!(state.is_accepting());
    }
}
