//! Independent final-API semantic review. No benchmarks or internal engine APIs.
//! All reference languages are compiled directly from fully bound descriptions.
use glrmask::{BuildOptions, Constraint, Grammar, Module, Optimization, Vocab};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
        (3, b"y".to_vec()), (4, b"xay".to_vec()), (5, b"ab".to_vec()),
        (7, b"a".to_vec()), (8, b"!".to_vec()), (9, b"z".to_vec()),
        (10, b"xabzy".to_vec()), (11, b"xy".to_vec()), (33, b"a".to_vec()),
        (63, b"<mark>".to_vec()), (127, b"<eos>".to_vec()),
    ])
}
fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|w| w & (1 << (id % 32)) != 0)
}
fn compare_prefixes(actual: &Constraint, expected: &Constraint, ids: &[u32], depth: usize) {
    let mut pending = vec![Vec::new()];
    while let Some(prefix) = pending.pop() {
        let mut a = actual.start();
        let mut e = expected.start();
        for &id in &prefix {
            a.commit_token(id).unwrap();
            e.commit_token(id).unwrap();
        }
        assert_eq!(a.mask(), e.mask(), "mask differs after {prefix:?}");
        assert_eq!(a.is_accepting(), e.is_accepting(), "acceptance differs after {prefix:?}");
        assert_eq!(a.is_rejected(), e.is_rejected(), "rejection differs after {prefix:?}");
        if prefix.len() < depth && !a.is_rejected() {
            for &id in ids {
                // Only advance prefixes admitted by the reference. Mask comparison
                // already checks rejection of every other ID. Invalid commits may
                // either return Err or enter a rejected state in legacy backends.
                if allowed(&e.mask(), id) {
                    let mut next = prefix.clone(); next.push(id); pending.push(next);
                }
            }
        }
    }
}
const TOKEN: &str = "glrm 1; start start; extern token MARK; nt start = MARK;";
const OUTER: &str = "glrm 1; start start; extern grammar middle; nt start = \"x\" middle \"y\";";
const BOTH: &str = "glrm 1; start start; extern token MARK; extern grammar child; nt start = \"x\" MARK child \"y\";";

#[test]
fn review_exact_value_validates_whole_mapping_not_just_bound_id() {
    let v = vocab();
    let mut entries = v.iter().map(|(id, bytes)| (id, bytes.to_vec())).collect::<Vec<_>>();
    entries.iter_mut().find(|(id, _)| *id == 9).unwrap().1 = b"other".to_vec();
    let incompatible = Vocab::new(entries);
    let g = Grammar::from_glrm(TOKEN).bind("MARK", incompatible.token(7).unwrap()).unwrap();
    assert!(g.compile(&v).is_err());
    let m = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap();
    assert!(m.bind("MARK", incompatible.token(7).unwrap()).is_err());
    // A separately constructed identical mapping is compatible; allocation identity isn't semantics.
    let identical = Vocab::new(v.iter().map(|(id, b)| (id, b.to_vec())).collect());
    assert!(g.compile(&incompatible).is_ok());
    let bound = m.bind("MARK", identical.token(7).unwrap()).unwrap().link().unwrap();
    assert!(allowed(&bound.start().mask(), 7));
    assert!(!allowed(&bound.start().mask(), 1));
}

#[test]
fn review_source_embedding_cannot_silently_close_open_token_module() {
    let v = vocab();
    let open = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap();
    let desc = Grammar::from_glrm(OUTER).bind("middle", &open).unwrap();
    assert!(desc.compile(&v).is_err(), "open child token slot must propagate to final-root validation");
}

#[test]
fn review_nested_open_token_module_links_after_loading() {
    let v = vocab();
    let open = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap();
    let top = Grammar::from_glrm(OUTER).compile_module(&v).unwrap();
    let nested = top.bind("middle", &open).unwrap();
    assert!(nested.link().is_err());
    let loaded = Module::load(nested.save()).unwrap();
    assert!(loaded.link().is_err());
    let actual = loaded.bind("middle.MARK", v.tokens([7, 33, 63]).unwrap()).unwrap().link().unwrap();
    let reference = Grammar::from_glrm(OUTER)
        .bind("middle", Grammar::from_glrm(TOKEN).bind("MARK", v.tokens([7, 33, 63]).unwrap()).unwrap())
        .unwrap().compile(&v).unwrap();
    compare_prefixes(&actual, &reference, &[0, 1, 3, 4, 7, 33, 63], 3);
}

#[test]
fn review_source_compiled_module_retains_open_token_metadata() {
    let v = vocab();
    let child = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap();
    let open = Grammar::from_glrm(OUTER).bind("middle", &child).unwrap().compile_module(&v).unwrap();
    assert!(open.link().is_err());
    let bound = Module::load(open.save()).unwrap().bind("middle.MARK", v.token(7).unwrap()).unwrap().link().unwrap();
    let expected = Grammar::from_glrm(OUTER)
        .bind("middle", Grammar::from_glrm(TOKEN).bind("MARK", v.token(7).unwrap()).unwrap())
        .unwrap().compile(&v).unwrap();
    compare_prefixes(&bound, &expected, &[0, 1, 3, 4, 7], 3);
}

#[test]
fn review_token_and_grammar_binding_order_and_roundtrips_agree() {
    let v = vocab();
    let source = Grammar::from_glrm(BOTH);
    let host = source.compile_module(&v).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "b""#).compile(&v).unwrap();
    let expected = source.bind("MARK", v.tokens([7, 63]).unwrap()).unwrap()
        .bind("child", &child).unwrap().compile(&v).unwrap();
    let token_first = host.bind("MARK", v.tokens([7, 63]).unwrap()).unwrap();
    let grammar_first = host.bind("child", &child).unwrap();
    let a = Module::load(token_first.save()).unwrap().bind("child", &child).unwrap().link().unwrap();
    let b = Module::load(grammar_first.save()).unwrap().bind("MARK", v.tokens([7, 63]).unwrap()).unwrap().link().unwrap();
    compare_prefixes(&a, &expected, &[0, 1, 2, 3, 4, 7, 63], 4);
    compare_prefixes(&b, &expected, &[0, 1, 2, 3, 4, 7, 63], 4);
    assert!(host.link().is_err(), "original module must remain open");
}

#[test]
fn review_multiple_exact_slots_remain_distinct_after_partial_save() {
    let v = vocab();
    let source = Grammar::from_glrm("glrm 1; start start; extern token A; extern token B; nt start = A B;");
    let original = source.compile_module(&v).unwrap();
    let partial = original.bind("A", v.token(7).unwrap()).unwrap();
    assert!(partial.link().is_err());
    let loaded = Module::load(partial.save()).unwrap();
    assert!(loaded.bind("A", v.token(33).unwrap()).is_err());
    let actual = loaded.bind("B", v.tokens([33, 63]).unwrap()).unwrap().link().unwrap();
    let expected = source.bind("A", v.token(7).unwrap()).unwrap()
        .bind("B", v.tokens([33, 63]).unwrap()).unwrap().compile(&v).unwrap();
    compare_prefixes(&actual, &expected, &[1, 7, 8, 33, 63], 3);
}

#[test]
fn review_bound_exact_child_can_be_reloaded_and_recomposed() {
    let v = vocab();
    let child = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap()
        .bind("MARK", v.tokens([7, 63]).unwrap()).unwrap().link().unwrap();
    let loaded = Constraint::load(child.save()).unwrap();
    let actual = Grammar::from_glrm(OUTER).bind("middle", &loaded).unwrap().compile(&v).unwrap();
    let expected = Grammar::from_glrm(OUTER)
        .bind("middle", Grammar::from_glrm(TOKEN).bind("MARK", v.tokens([7, 63]).unwrap()).unwrap())
        .unwrap().compile(&v).unwrap();
    compare_prefixes(&actual, &expected, &[0, 1, 3, 4, 7, 63], 3);
}

#[test]
fn review_open_module_artifact_cannot_be_loaded_as_runnable_constraint() {
    let v = vocab();
    for source in [OUTER, TOKEN, BOTH] {
        let module = Grammar::from_glrm(source).compile_module(&v).unwrap();
        assert!(Constraint::load(module.save()).is_err(), "unbound module loaded as runnable: {source}");
    }
}

#[test]
fn review_truncated_module_artifacts_fail_without_panics() {
    let v = vocab();
    let module = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap();
    let bytes = module.save();
    for len in [0, 1, 4, 8, 16, bytes.len() / 2, bytes.len() - 1] {
        let result = std::panic::catch_unwind(|| Module::load(bytes[..len].to_vec()));
        assert!(result.is_ok(), "panic loading truncated artifact of {len} bytes");
        assert!(result.unwrap().is_err(), "accepted truncated artifact of {len} bytes");
    }
}

#[test]
fn review_compiled_binding_rejects_wrong_kind_and_duplicate_names() {
    let v = vocab();
    let module = Grammar::from_glrm(BOTH).compile_module(&v).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "b""#).compile(&v).unwrap();
    assert!(module.bind("MARK", &child).is_err());
    assert!(module.bind("child", v.token(7).unwrap()).is_err());
    assert!(module.bind("absent", &child).is_err());
    assert!(module.bind("absent", v.token(7).unwrap()).is_err());
    let once = module.bind("MARK", v.token(7).unwrap()).unwrap();
    assert!(once.bind("MARK", v.token(33).unwrap()).is_err());
    let once = module.bind("child", &child).unwrap();
    assert!(once.bind("child", &child).is_err());
}

#[test]
fn review_repeated_same_open_child_keeps_per_slot_bindings_independent() {
    let v = vocab();
    let outer = Grammar::from_glrm("glrm 1; start start; extern grammar left; extern grammar right; nt start = left right;");
    let child = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap();
    let open = outer.compile_module(&v).unwrap().bind("left", &child).unwrap().bind("right", &child).unwrap();
    let actual = Module::load(open.save()).unwrap()
        .bind("left.MARK", v.token(7).unwrap()).unwrap()
        .bind("right.MARK", v.token(33).unwrap()).unwrap().link().unwrap();
    let expected = outer.bind("left", Grammar::from_glrm(TOKEN).bind("MARK", v.token(7).unwrap()).unwrap()).unwrap()
        .bind("right", Grammar::from_glrm(TOKEN).bind("MARK", v.token(33).unwrap()).unwrap()).unwrap()
        .compile(&v).unwrap();
    compare_prefixes(&actual, &expected, &[1, 7, 33], 3);
    assert!(child.link().is_err());
}

#[test]
fn review_one_token_crosses_parent_child_return_and_sibling() {
    let v = vocab();
    let left = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    let right = Grammar::from_ebnf(r#"start ::= "z""#).compile_module(&v).unwrap();
    let parent = Grammar::from_glrm("glrm 1; start start; extern grammar left; extern grammar right; nt start = \"x\" left \"b\" right \"y\";");
    let actual = parent.bind("left", &left).unwrap().bind("right", &right).unwrap().compile(&v).unwrap();
    let expected = Grammar::from_ebnf(r#"start ::= "xabzy""#).compile(&v).unwrap();
    assert!(allowed(&actual.start().mask(), 10));
    compare_prefixes(&actual, &expected, &[0, 1, 2, 3, 5, 9, 10], 3);
}

#[test]
fn review_nullable_child_preserves_cross_boundary_mask_after_load() {
    let v = vocab();
    let nullable = Grammar::from_glrm("glrm 1; start start; nt start = eps;").compile(&v).unwrap();
    let m = Grammar::from_glrm(OUTER).compile_module(&v).unwrap().bind("middle", &nullable).unwrap();
    let actual = Module::load(m.save()).unwrap().link().unwrap();
    let expected = Grammar::from_ebnf(r#"start ::= "xy""#).compile(&v).unwrap();
    assert!(allowed(&actual.start().mask(), 11));
    compare_prefixes(&actual, &expected, &[0, 1, 3, 4, 11], 3);
}

#[test]
fn review_end_tokens_are_root_only_and_survive_root_roundtrip() {
    let v = vocab();
    let child = Grammar::from_ebnf(r#"start ::= "a""#)
        .compile_with(&v, BuildOptions::default().end_tokens([127])).unwrap();
    let loaded = Constraint::load(child.save()).unwrap();
    for standalone in [&child, &loaded] {
        let mut state = standalone.start();
        assert!(!allowed(&state.mask(), 127));
        state.commit_token(1).unwrap();
        assert!(state.is_accepting());
        assert!(allowed(&state.mask(), 127));
        state.commit_token(127).unwrap();
        assert!(state.mask().iter().all(|w| *w == 0));
    }
    let actual = Grammar::from_glrm(OUTER).bind("middle", &loaded).unwrap()
        .compile_with(&v, BuildOptions::default().end_tokens([63])).unwrap();
    let mut state = actual.start();
    assert!(allowed(&state.mask(), 4), "embedding imports body, not the child's EOS language");
    state.commit_token(0).unwrap(); state.commit_token(1).unwrap();
    assert!(!allowed(&state.mask(), 127));
    assert!(!allowed(&state.mask(), 63));
    assert!(allowed(&state.mask(), 3));
    state.commit_token(3).unwrap();
    assert!(!allowed(&state.mask(), 127));
    assert!(allowed(&state.mask(), 63));
}

#[test]
fn review_root_end_policy_does_not_erase_grammar_level_exact_tokens() {
    let v = vocab();
    let child = Grammar::from_glrm(TOKEN).bind("MARK", v.token(63).unwrap()).unwrap()
        .compile_with(&v, BuildOptions::default().end_tokens([127])).unwrap();
    let actual = Grammar::from_glrm(OUTER).bind("middle", &child).unwrap().compile(&v).unwrap();
    let mut state = actual.start();
    state.commit_token(0).unwrap();
    assert!(allowed(&state.mask(), 63), "explicit grammar-level token is retained when root EOS is removed");
    assert!(!allowed(&state.mask(), 127));
    state.commit_token(63).unwrap(); state.commit_token(3).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn review_optimization_choices_preserve_semantics_with_exact_bindings() {
    let v = vocab();
    let child = Grammar::from_ebnf(r#"start ::= "b""#).compile(&v).unwrap();
    let desc = Grammar::from_glrm(BOTH).bind("MARK", v.tokens([7, 63]).unwrap()).unwrap().bind("child", &child).unwrap();
    let baseline = desc.compile(&v).unwrap();
    for mode in [Optimization::Auto, Optimization::FastBuild, Optimization::FastRuntime] {
        let actual = desc.compile_with(&v, BuildOptions::default().optimization(mode)).unwrap();
        compare_prefixes(&actual, &baseline, &[0, 1, 2, 3, 5, 7, 63], 4);
    }
}

#[test]
fn review_parent_and_child_open_tokens_have_distinct_private_placeholders() {
    let v = vocab();
    let parent = Grammar::from_glrm("glrm 1; start start; extern token MARK; extern grammar child; nt start = MARK child;");
    let child = Grammar::from_glrm(TOKEN).compile_module(&v).unwrap();
    // Both component compilers independently choose the same first unused ID.
    // Public linking must use typed slot/terminal identity, not token-ID identity.
    let open = parent.bind("child", &child).unwrap().compile_module(&v).unwrap();
    let actual = Module::load(open.save()).unwrap()
        .bind("MARK", v.token(7).unwrap()).unwrap()
        .bind("child.MARK", v.token(33).unwrap()).unwrap().link().unwrap();
    let expected = parent.bind("MARK", v.token(7).unwrap()).unwrap()
        .bind("child", Grammar::from_glrm(TOKEN).bind("MARK", v.token(33).unwrap()).unwrap()).unwrap()
        .compile(&v).unwrap();
    compare_prefixes(&actual, &expected, &[1, 7, 33], 3);
}

#[test]
fn review_empty_byte_exact_tokens_preserve_mask_and_commit_on_valid_prefixes() {
    let mut entries = vocab().iter().map(|(id, b)| (id, b.to_vec())).collect::<Vec<_>>();
    for (id, bytes) in &mut entries {
        if *id == 63 { bytes.clear(); }
    }
    let v = Vocab::new(entries);
    let source = Grammar::from_glrm("glrm 1; start start; extern token MARK; nt start = \"x\" MARK \"y\";");
    let expected = source.bind("MARK", v.tokens([7, 63]).unwrap()).unwrap().compile(&v).unwrap();
    let actual = source.compile_module(&v).unwrap().bind("MARK", v.tokens([7, 63]).unwrap()).unwrap().link().unwrap();
    compare_prefixes(&actual, &expected, &[0, 1, 3, 4, 7, 63], 3);
}
