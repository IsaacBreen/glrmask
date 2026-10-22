//! Independent final-API semantic review. No benchmarks or internal engine APIs.
//! All reference languages are compiled directly from fully bound descriptions.
use glrmask::{BuildOptions, Constraint, Grammar, PreparedGrammar, MaskStrategy, Vocab};

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
fn review_token_validates_membership_without_claiming_vocabulary_identity() {
    let v = vocab();
    let mut entries = v.iter().map(|(id, bytes)| (id, bytes.to_vec())).collect::<Vec<_>>();
    entries.iter_mut().find(|(id, _)| *id == 7).unwrap().1 = b"other".to_vec();
    let different = Vocab::new(entries);
    let token = glrmask::Token::new(7);
    let g = Grammar::from_glrm(TOKEN).bind_snapshot("MARK", token).unwrap();
    for target in [&v, &different] {
        let c = g.compile(target).unwrap();
        assert!(allowed(&c.start().mask(), 7));
        assert!(!allowed(&c.start().mask(), 1));
    }
    let m = Grammar::from_glrm(TOKEN).prepare(&v).unwrap();
    assert!(m.bind_snapshot("MARK", glrmask::Token::new(6)).is_err());
    assert!(m.bind_snapshot("MARK", glrmask::Token::new(u32::MAX)).is_err());
    assert!(m.bind_snapshot("MARK", token).unwrap().link().is_ok());
}

#[test]
fn review_source_embedding_cannot_silently_close_open_token_module() {
    let v = vocab();
    let open_source = Grammar::from_glrm(TOKEN);
    let open = open_source.prepare(&v).unwrap();
    assert!(open.link().is_err());
    let desc = Grammar::from_glrm(OUTER).bind_snapshot("middle", &open_source).unwrap();
    assert!(desc.compile(&v).is_err(), "open child token slot must propagate to final-root validation");
}

#[test]
fn review_nested_open_token_module_links_after_loading() {
    let v = vocab();
    let open_source = Grammar::from_glrm(TOKEN);
    let top = Grammar::from_glrm(OUTER).bind_snapshot("middle", &open_source).unwrap();
    let nested = top.prepare(&v).unwrap();
    assert!(nested.link().is_err());
    let loaded = PreparedGrammar::load(nested.save()).unwrap();
    assert!(loaded.link().is_err());
    let actual = loaded.bind_snapshot("middle::MARK", v.tokens([7, 33, 63]).unwrap()).unwrap().link().unwrap();
    let reference = Grammar::from_glrm(OUTER)
        .bind_snapshot("middle", Grammar::from_glrm(TOKEN).bind_snapshot("MARK", v.tokens([7, 33, 63]).unwrap()).unwrap())
        .unwrap().compile(&v).unwrap();
    compare_prefixes(&actual, &reference, &[0, 1, 3, 4, 7, 33, 63], 3);
}

#[test]
fn review_source_compiled_module_retains_open_token_metadata() {
    let v = vocab();
    let child = Grammar::from_glrm(TOKEN);
    let open = Grammar::from_glrm(OUTER).bind_snapshot("middle", &child).unwrap().prepare(&v).unwrap();
    assert!(open.link().is_err());
    let bound = PreparedGrammar::load(open.save()).unwrap().bind_snapshot("middle::MARK", glrmask::Token::new(7)).unwrap().link().unwrap();
    let expected = Grammar::from_glrm(OUTER)
        .bind_snapshot("middle", Grammar::from_glrm(TOKEN).bind_snapshot("MARK", glrmask::Token::new(7)).unwrap())
        .unwrap().compile(&v).unwrap();
    compare_prefixes(&bound, &expected, &[0, 1, 3, 4, 7], 3);
}

#[test]
fn review_token_and_grammar_binding_order_and_roundtrips_agree() {
    let v = vocab();
    let source = Grammar::from_glrm(BOTH);
    let host = source.prepare(&v).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "b""#).prepare(&v).unwrap();
    let expected = source.bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap()
        .bind_snapshot("child", Grammar::from_ebnf(r#"start ::= "b""#)).unwrap().compile(&v).unwrap();
    let token_first = host.bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap();
    let grammar_first = host.bind_snapshot("child", &child).unwrap();
    let a = PreparedGrammar::load(token_first.save()).unwrap().bind_snapshot("child", &child).unwrap().link().unwrap();
    let b = PreparedGrammar::load(grammar_first.save()).unwrap().bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap().link().unwrap();
    compare_prefixes(&a, &expected, &[0, 1, 2, 3, 4, 7, 63], 4);
    compare_prefixes(&b, &expected, &[0, 1, 2, 3, 4, 7, 63], 4);
    assert!(host.link().is_err(), "original module must remain open");
}

#[test]
fn review_multiple_exact_slots_remain_distinct_after_partial_save() {
    let v = vocab();
    let source = Grammar::from_glrm("glrm 1; start start; extern token A; extern token B; nt start = A B;");
    let original = source.prepare(&v).unwrap();
    let partial = original.bind_snapshot("A", glrmask::Token::new(7)).unwrap();
    assert!(partial.link().is_err());
    let loaded = PreparedGrammar::load(partial.save()).unwrap();
    assert!(loaded.bind_snapshot("A", glrmask::Token::new(33)).is_err());
    let actual = loaded.bind_snapshot("B", v.tokens([33, 63]).unwrap()).unwrap().link().unwrap();
    let expected = source.bind_snapshot("A", glrmask::Token::new(7)).unwrap()
        .bind_snapshot("B", v.tokens([33, 63]).unwrap()).unwrap().compile(&v).unwrap();
    compare_prefixes(&actual, &expected, &[1, 7, 8, 33, 63], 3);
}

#[test]
fn review_bound_exact_child_can_be_reloaded_and_recomposed() {
    let v = vocab();
    let child = Grammar::from_glrm(TOKEN).prepare(&v).unwrap()
        .bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap();
    let loaded = PreparedGrammar::load(child.save()).unwrap();
    let actual = Grammar::from_glrm(OUTER).prepare(&v).unwrap().bind_snapshot("middle", &loaded).unwrap().link().unwrap();
    let expected = Grammar::from_glrm(OUTER)
        .bind_snapshot("middle", Grammar::from_glrm(TOKEN).bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap())
        .unwrap().compile(&v).unwrap();
    compare_prefixes(&actual, &expected, &[0, 1, 3, 4, 7, 63], 3);
}

#[test]
fn review_open_module_artifact_cannot_be_loaded_as_runnable_constraint() {
    let v = vocab();
    for source in [OUTER, TOKEN, BOTH] {
        let module = Grammar::from_glrm(source).prepare(&v).unwrap();
        assert!(Constraint::load(module.save()).is_err(), "unbound module loaded as runnable: {source}");
    }
}

#[test]
fn review_truncated_module_artifacts_fail_without_panics() {
    let v = vocab();
    let module = Grammar::from_glrm(TOKEN).prepare(&v).unwrap();
    let bytes = module.save();
    for len in [0, 1, 4, 8, 16, bytes.len() / 2, bytes.len() - 1] {
        let result = std::panic::catch_unwind(|| PreparedGrammar::load(bytes[..len].to_vec()));
        assert!(result.is_ok(), "panic loading truncated artifact of {len} bytes");
        assert!(result.unwrap().is_err(), "accepted truncated artifact of {len} bytes");
    }
}

#[test]
fn review_prepared_binding_checks_kinds_and_rejects_duplicate_names() {
    let v = vocab();
    let module = Grammar::from_glrm(BOTH).prepare(&v).unwrap();
    let child = Grammar::from_ebnf(r#"start ::= "b""#).prepare(&v).unwrap();
    assert!(module.bind_snapshot("MARK", &child).is_err());
    let widened=module.bind_snapshot("child",glrmask::Token::new(7)).unwrap();
    assert_eq!(widened.requirements().unwrap().iter().map(|r|r.name()).collect::<Vec<_>>(),["MARK"]);
    assert!(widened.bind_snapshot("child",glrmask::Token::new(33)).is_err());
    assert_eq!(module.requirements().unwrap().len(),2);
    assert!(module.bind_snapshot("absent", &child).is_err());
    assert!(module.bind_snapshot("absent", glrmask::Token::new(7)).is_err());
    let once = module.bind_snapshot("MARK", glrmask::Token::new(7)).unwrap();
    assert!(once.bind_snapshot("MARK", glrmask::Token::new(33)).is_err());
    assert!(once.bind_snapshot("MARK", &child).is_err());
    let once = module.bind_snapshot("child", &child).unwrap();
    assert!(once.bind_snapshot("child", &child).is_err());
}

#[test]
fn review_repeated_same_open_child_keeps_per_slot_bindings_independent() {
    let v = vocab();
    let outer = Grammar::from_glrm("glrm 1; start start; extern grammar left; extern grammar right; nt start = left right;");
    let child = Grammar::from_glrm(TOKEN);
    let child_open = child.prepare(&v).unwrap();
    let open = outer.bind_snapshot("left", &child).unwrap().bind_snapshot("right", &child).unwrap()
        .prepare(&v).unwrap();
    let actual = PreparedGrammar::load(open.save()).unwrap()
        .bind_snapshot("left::MARK", glrmask::Token::new(7)).unwrap()
        .bind_snapshot("right::MARK", glrmask::Token::new(33)).unwrap().link().unwrap();
    let expected = outer.bind_snapshot("left", Grammar::from_glrm(TOKEN).bind_snapshot("MARK", glrmask::Token::new(7)).unwrap()).unwrap()
        .bind_snapshot("right", Grammar::from_glrm(TOKEN).bind_snapshot("MARK", glrmask::Token::new(33)).unwrap()).unwrap()
        .compile(&v).unwrap();
    compare_prefixes(&actual, &expected, &[1, 7, 33], 3);
    assert!(child_open.link().is_err());
}

#[test]
fn review_one_token_crosses_parent_child_return_and_sibling() {
    let v = vocab();
    let left = Grammar::from_ebnf(r#"start ::= "a""#).prepare(&v).unwrap();
    let right = Grammar::from_ebnf(r#"start ::= "z""#).prepare(&v).unwrap();
    let parent = Grammar::from_glrm("glrm 1; start start; extern grammar left; extern grammar right; nt start = \"x\" left \"b\" right \"y\";");
    let actual = parent.prepare(&v).unwrap().bind_snapshot("left", &left).unwrap().bind_snapshot("right", &right).unwrap().link().unwrap();
    let expected = Grammar::from_ebnf(r#"start ::= "xabzy""#).compile(&v).unwrap();
    assert!(allowed(&actual.start().mask(), 10));
    compare_prefixes(&actual, &expected, &[0, 1, 2, 3, 5, 9, 10], 3);
}

#[test]
fn review_nullable_child_preserves_cross_boundary_mask_after_load() {
    let v = vocab();
    let nullable = Grammar::from_glrm("glrm 1; start start; nt start = eps;").prepare(&v).unwrap();
    let m = Grammar::from_glrm(OUTER).prepare(&v).unwrap().bind_snapshot("middle", &nullable).unwrap();
    let actual = PreparedGrammar::load(m.save()).unwrap().link().unwrap();
    let expected = Grammar::from_ebnf(r#"start ::= "xy""#).compile(&v).unwrap();
    assert!(allowed(&actual.start().mask(), 11));
    compare_prefixes(&actual, &expected, &[0, 1, 3, 4, 11], 3);
}

#[test]
fn review_end_tokens_are_root_only_and_survive_root_roundtrip() {
    let v = vocab();
    let prepared = Grammar::from_ebnf(r#"start ::= "a""#).prepare(&v).unwrap();
    let child = prepared.link_with(BuildOptions::default().end_tokens([127])).unwrap();
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
    let actual = Grammar::from_glrm(OUTER).prepare(&v).unwrap().bind_snapshot("middle", &PreparedGrammar::load(prepared.save()).unwrap()).unwrap()
        .link_with(BuildOptions::default().end_tokens([63])).unwrap();
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
    let child = Grammar::from_glrm(TOKEN).bind_snapshot("MARK", glrmask::Token::new(63)).unwrap()
        .prepare(&v).unwrap();
    let standalone = child.link_with(BuildOptions::default().end_tokens([127])).unwrap();
    assert!(allowed(&standalone.start().mask(),63));
    let actual = Grammar::from_glrm(OUTER).prepare(&v).unwrap().bind_snapshot("middle", &child).unwrap().link().unwrap();
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
    let child = Grammar::from_ebnf(r#"start ::= "b""#).prepare(&v).unwrap();
    let desc = Grammar::from_glrm(BOTH).bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap()
        .prepare(&v).unwrap().bind_snapshot("child", &child).unwrap();
    let baseline = Grammar::from_glrm(BOTH).bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap()
        .bind_snapshot("child", Grammar::from_ebnf(r#"start ::= "b""#)).unwrap().compile(&v).unwrap();
    for mode in [MaskStrategy::Auto, MaskStrategy::TokenTrie, MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let actual = desc.link_with(BuildOptions::default().optimization(mode)).unwrap();
        compare_prefixes(&actual, &baseline, &[0, 1, 2, 3, 5, 7, 63], 4);
    }
}

#[test]
fn review_parent_and_child_open_tokens_have_distinct_private_placeholders() {
    let v = vocab();
    let parent = Grammar::from_glrm("glrm 1; start start; extern token MARK; extern grammar child; nt start = MARK child;");
    let child = Grammar::from_glrm(TOKEN);
    // Both component compilers independently choose the same first unused ID.
    // Public linking must use typed slot/terminal identity, not token-ID identity.
    let open = parent.bind_snapshot("child", &child).unwrap().prepare(&v).unwrap();
    let actual = PreparedGrammar::load(open.save()).unwrap()
        .bind_snapshot("MARK", glrmask::Token::new(7)).unwrap()
        .bind_snapshot("child::MARK", glrmask::Token::new(33)).unwrap().link().unwrap();
    let expected = parent.bind_snapshot("MARK", glrmask::Token::new(7)).unwrap()
        .bind_snapshot("child", Grammar::from_glrm(TOKEN).bind_snapshot("MARK", glrmask::Token::new(33)).unwrap()).unwrap()
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
    let expected = source.bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap().compile(&v).unwrap();
    let actual = source.prepare(&v).unwrap().bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap().link().unwrap();
    compare_prefixes(&actual, &expected, &[0, 1, 3, 4, 7, 63], 3);
}

#[test]
fn review_repeated_roundtrips_preserve_distinct_packed_acceptance_rows() {
    let mut entries = vocab().iter().map(|(id, b)| (id, b.to_vec())).collect::<Vec<_>>();
    for (id, bytes) in &mut entries {
        if *id == 63 { bytes.clear(); }
    }
    let v = Vocab::new(entries);
    let source = Grammar::from_glrm("glrm 1; start start; extern token MARK; nt start = MARK \"b\" | \"a\" \"y\";");
    let expected = source.bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap().compile(&v).unwrap();
    // The parent is packed before the exact-token adapter is linked. A second
    // serialization must preserve each packed pool ID, not deduplicate the
    // intentionally-empty structural Weight placeholders.
    let open = PreparedGrammar::load(source.prepare(&v).unwrap().save()).unwrap();
    let mut module = open.bind_snapshot("MARK", v.tokens([7, 63]).unwrap()).unwrap();
    for _ in 0..4 {
        module = PreparedGrammar::load(module.save()).unwrap();
        let mut actual = module.link().unwrap();
        compare_prefixes(&actual, &expected, &[1, 2, 3, 5, 7, 63], 3);
        for _ in 0..3 {
            actual = Constraint::load(actual.save()).unwrap();
            compare_prefixes(&actual, &expected, &[1, 2, 3, 5, 7, 63], 3);
        }
    }
}

#[test]
fn review_boundary_request_is_eager_and_preserves_semantics() {
    use glrmask::BoundaryTriggerDetail;
    let v = vocab();
    let grammar = Grammar::from_ebnf(r#"start ::= "a" "b"?"#);
    for mode in [MaskStrategy::Auto, MaskStrategy::TokenTrie, MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
        let options = BuildOptions::default().optimization(mode);
        let plain = grammar.compile_with_build_options(&v, options.clone()).unwrap();
        let none = grammar.compile_with_build_options(&v, options.clone().boundary_trigger(BoundaryTriggerDetail::None)).unwrap();
        assert_ne!(plain.save(), none.save());
        for detail in [BoundaryTriggerDetail::TokenSet, BoundaryTriggerDetail::Exact] {
            if mode == MaskStrategy::TokenTrie && detail == BoundaryTriggerDetail::Exact {
                assert!(grammar.compile_with_build_options(&v, options.clone().boundary_trigger(detail)).unwrap_err().to_string().contains("retained-LR Dynamic"));
                continue;
            }
            let requested = grammar.compile_with_build_options(&v, options.clone().boundary_trigger(detail)).unwrap();
            // Inspect persistence before start/first mask: trigger construction is eager.
            let bytes = requested.save();
            if detail == BoundaryTriggerDetail::TokenSet {
                assert_eq!(bytes, plain.save(), "default TokenSet must match an explicit request");
            } else {
                assert_ne!(bytes, plain.save());
            }
            compare_prefixes(&requested, &plain, &[1, 2, 5, 7, 33], 3);
            assert_eq!(requested.save(), bytes, "matching must not materialize trigger metadata");
            compare_prefixes(&Constraint::load(bytes).unwrap(), &plain, &[1, 2, 5, 7, 33], 3);
        }
    }
}

#[test]
fn review_deferred_boundary_request_roundtrips_and_link_override_wins() {
    use glrmask::BoundaryTriggerDetail;
    let v = vocab();
    let grammar = Grammar::from_glrm(OUTER);
    let plain = grammar.prepare(&v).unwrap();
    let explicit_none = grammar.prepare_with(&v,
        BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::None)).unwrap();
    // Independent compilations need not assign identical automaton state IDs or
    // serialize to canonical bytes. Check the public language/requirements here;
    // public_api::final_boundary_option_tests directly asserts the stored
    // None/Tokens/Exact variants and their scoping before and after linking.
    assert_eq!(plain.requirements().unwrap(), explicit_none.requirements().unwrap());
    let child = Grammar::from_ebnf(r#"start ::= "a" "b"?"#).prepare(&v).unwrap();
    let default_link = plain.bind_snapshot("middle", &child).unwrap().link().unwrap();
    let explicit_none_link = PreparedGrammar::load(explicit_none.save()).unwrap()
        .bind_snapshot("middle", &child).unwrap().link().unwrap();
    compare_prefixes(&explicit_none_link, &default_link, &[0, 1, 2, 3, 4, 5, 10, 11], 4);
    for detail in [BoundaryTriggerDetail::TokenSet, BoundaryTriggerDetail::Exact] {
        let module = grammar.prepare_with(&v, BuildOptions::default().boundary_trigger(detail)).unwrap();
        assert!(module.link().is_err(), "trigger request cannot bypass unresolved slots");
        let saved = module.save();
        assert!(saved.starts_with(b"GLRMOD"));
        let loaded = PreparedGrammar::load(saved.clone()).unwrap();
        assert_eq!(loaded.save(), saved);
        let bound = loaded.bind_snapshot("middle", &child).unwrap();
        let plain_loaded = PreparedGrammar::load(plain.save()).unwrap();
        let reference = plain_loaded.bind_snapshot("middle", &child).unwrap().link().unwrap();
        let inherited = bound.link().unwrap();
        compare_prefixes(&inherited, &reference, &[0, 1, 2, 3, 4, 5, 10, 11], 4);
        let suppressed = bound.link_with(BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::None)).unwrap();
        compare_prefixes(&suppressed, &reference, &[0, 1, 2, 3, 4, 5, 10, 11], 4);
        let tokens = bound.link_with(BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::TokenSet)).unwrap();
        let tokens_reference = plain_loaded.bind_snapshot("middle", &child).unwrap().link_with(
            BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::TokenSet)).unwrap();
        compare_prefixes(&tokens, &tokens_reference, &[0, 1, 2, 3, 4, 5, 10, 11], 4);
        let bound_loaded = PreparedGrammar::load(bound.save()).unwrap().link().unwrap();
        compare_prefixes(&bound_loaded, &inherited, &[0, 1, 2, 3, 4, 5, 10, 11], 4);
        let mut invalid = saved;
        invalid[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(PreparedGrammar::load(invalid).is_err());
    }
}
