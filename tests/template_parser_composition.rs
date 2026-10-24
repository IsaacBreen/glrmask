use glrmask::test_modes::MaskMode;
use std::collections::BTreeSet;
use glrmask::{BuildOptions, Constraint, Grammar, ParserBackend, TokenSet, Vocab};

const HOST: &str = r#"glrm 1; start root; extern grammar child; nt root = "x" child "y";"#;

#[test]
fn nullable_lexical_source_is_preserved_by_every_compiled_child_backend() {
    let (vocab, tokens) = vocabulary();
    let sources = [
        Grammar::from_glrm("start root; t A = /a?/; nt root = A;").unwrap(),
        Grammar::from_glrm("start root; t A = /a?/ & /a*/; nt root = A;").unwrap(),
        // EBNF has no slash-delimited regex syntax; its explicit optional
        // literal is the independent parser-nullability control case.
        Grammar::from_ebnf(r#"start ::= "a"?"#).unwrap(),
        Grammar::from_lark("start: A\nA: /a?/").unwrap(),
    ];
    let words: &[&[u8]] = &[b"xy", b"xay"];
    for (source_index, source) in sources.iter().enumerate() {
        for backend in [ParserBackend::TemplateDfa] {
            for child_mode in [MaskMode::C, MaskMode::ParserDwa] {
                let child = source.compile_with_build_options(&vocab,
                    child_mode.build_options().parser_backend(backend)).unwrap();
                let saved_child = Constraint::load(child.save(), &vocab).unwrap();
                for child in [&child, &saved_child] {
                    let bound = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap()
                        .bind_snapshot("child", child).unwrap();
                    for link_mode in [MaskMode::C, MaskMode::ParserDwa] {
                        let linked = bound.link_with(
                            link_mode.build_options().parser_backend(backend)).unwrap();
                        let loaded = Constraint::load(linked.save(), &vocab).unwrap();
                        let external = Some(Constraint::load(
                            linked.save(), &vocab).unwrap());
                        for c in [&linked, &loaded].into_iter().chain(external.as_ref()) {
                            for prefix in [b"".as_slice(), b"x", b"xy", b"xa", b"xay"] {
                                let mut state = c.start(); assert!(state.commit_bytes(prefix));
                                assert_eq!(state.is_accepting(), words.contains(&prefix));
                                let mask = state.mask();
                                for (id, bytes) in &tokens {
                                    let mut word = prefix.to_vec(); word.extend(bytes);
                                    let expected = words.iter().any(|candidate| candidate.starts_with(&word));
                                    let actual = mask.get(*id as usize / 32)
                                        .is_some_and(|bits| bits & (1 << (id % 32)) != 0);
                                    assert_eq!(actual, expected, "source={source_index} backend={backend:?} \
                                        child={child_mode:?} link={link_mode:?} prefix={prefix:?} token={bytes:?}");
                                    if actual {
                                        let mut branch = state.clone(); assert!(branch.commit_token(*id));
                                        assert_eq!(branch.is_accepting(), words.contains(&word.as_slice()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_regular_precompiled_children_keep_their_one_symbol_return_frame() {
    use glrmask::__private::{ConstraintExt, into_template_parser, parser_backend_report};
    use glrmask_grammar::__private::grammar::flat::{DirectRegularAutomaton, DirectRegularState,
        GrammarDef, Rule, Symbol, Terminal};
    use std::collections::BTreeMap;
    // The outer p C q wrapper must have actual crossing model tokens; without
    // them it correctly needs no B shard at that level. Keep both entering and
    // leaving crossings, and a whole token crossing both interfaces.
    let (_, mut tokens) = vocabulary();
    tokens.extend([(4000, b"px".to_vec()), (4001, b"yq".to_vec()),
        (4002, b"pxayq".to_vec()), (4003, b"pxyq".to_vec())]);
    let vocab = Vocab::new(tokens.clone());
    for nullable in [false, true] {
        let mut rules = vec![Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] }];
        if nullable { rules.push(Rule { lhs: 0, rhs: vec![] }); }
        let grammar = GrammarDef { rules, start: 0,
            terminals: vec![Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"b".to_vec() }],
            direct_regular_automaton: Some(DirectRegularAutomaton { start_states: vec![0], states: vec![
                DirectRegularState { is_accepting: nullable,
                    transitions: BTreeMap::from([(0, vec![1]), (1, vec![2])]), epsilons: vec![] },
                DirectRegularState { is_accepting: true,
                    transitions: BTreeMap::from([(1, vec![2])]), epsilons: vec![] },
                DirectRegularState { is_accepting: true, ..Default::default() },
            ] }), ..Default::default() };
        let source = Constraint::compile_grammar_def_json(&serde_json::to_string(&grammar).unwrap(), &vocab).unwrap();
        assert_eq!(parser_backend_report(&source)["sparse_regular"], true,
            "this test must exercise the sparse regular frontend, not an ordinary LR row");
        let words: &[&[u8]] = if nullable { &[b"xy", b"xay", b"xby", b"xaby"] }
            else { &[b"xay", b"xby", b"xaby"] };
        // A raw sparse child must select the native template linker as well;
        // the LR splicer requires augmented rules which this frontend omits.
        let legacy = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap().bind_snapshot("child", &source).unwrap();
        for mode in [MaskMode::C, MaskMode::ParserDwa] {
            let linked = legacy.link_with(mode.build_options()
                .parser_backend(ParserBackend::TemplateDfa)).unwrap();
            assert_language(&linked, &tokens, words);
            assert_language(&Constraint::load(linked.save(), &vocab).unwrap(), &tokens, words);
        }
        let child = into_template_parser(source).unwrap();
        assert_eq!(parser_backend_report(&child)["finite_embedding"], true);
        let child = Constraint::load(child.save(), &vocab).unwrap();
        let host = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap().bind_snapshot("child", &child).unwrap();
        for mode in [MaskMode::C, MaskMode::ParserDwa] {
            let c = host.link_with(mode.build_options()
                .parser_backend(ParserBackend::TemplateDfa)).unwrap();
            let loaded = Constraint::load(c.save(), &vocab).unwrap();
            let external = Constraint::load(c.save(), &vocab).unwrap();
            for c in [&c, &loaded, &external] { assert_language(c, &tokens, words); }
            let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "p" C "q";"#).unwrap()
                .prepare(&vocab).unwrap().bind_snapshot("C", &loaded).unwrap()
                .link_with(mode.build_options().parser_backend(ParserBackend::TemplateDfa)).unwrap();
            let nested_words = words.iter().map(|word| {
                let mut out = vec![b'p']; out.extend_from_slice(word); out.push(b'q'); out
            }).collect::<Vec<_>>();
            let nested_words = nested_words.iter().map(Vec::as_slice).collect::<Vec<_>>();
            assert_language(&outer, &tokens, &nested_words);
            assert_language(&Constraint::load(outer.save(), &vocab).unwrap(), &tokens, &nested_words);
            if mode == MaskMode::ParserDwa { assert_static_boundaries(&outer); }
        }
    }
}

#[test]
fn static_composition_with_no_crossing_tokens_needs_no_boundary_shards() {
    const CHILD: &str = "GLRMASK_TEMPLATE_STATIC_EMPTY_BOUNDARY_TEST";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("static_composition_with_no_crossing_tokens_needs_no_boundary_shards")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let tokens = b"xaypq".iter().enumerate().map(|(id, byte)| (id as u32, vec![*byte]))
        .collect::<Vec<_>>();
    let vocab = Vocab::new(tokens.clone());
    let options = || MaskMode::ParserDwa.build_options()
        .parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a""#).unwrap().compile_with_build_options(&vocab, options()).unwrap();
    let middle = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap()
        .bind_snapshot("child", &leaf).unwrap().link_with(options()).unwrap();
    let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar C; nt root = "p" C "q";"#).unwrap()
        .prepare(&vocab).unwrap().bind_snapshot("C", &middle).unwrap().link_with(options()).unwrap();
    fn check_empty_shards(report: &serde_json::Value) {
        assert_eq!(report["lr_table_present"], false);
        if let Some(children) = report["component_parsers"].as_array() {
            assert_eq!(report["static_boundary_shards"], 0, "{report}");
            assert_eq!(report["dynamic_boundary_shards"], 0, "{report}");
            assert_eq!(report["packed_lr_compiler_table_present"], false);
            for child in children { check_empty_shards(child); }
        }
    }
    for c in [&outer, &Constraint::load(outer.save(), &vocab).unwrap(),
        &Constraint::load(outer.save(), &vocab).unwrap()] {
        check_empty_shards(&glrmask::__private::parser_backend_report(c));
        assert_language(c, &tokens, &[b"pxayq"]);
    }
}

fn assert_static_boundaries(constraint: &Constraint) {
    fn check(report: &serde_json::Value) {
        assert_eq!(report["lr_table_present"], false);
        if let Some(children) = report["component_parsers"].as_array() {
            assert_eq!(report["dynamic_boundary_shards"], 0, "{report}");
            assert_eq!(report["packed_lr_compiler_table_present"], false);
            for child in children { check(child); }
        }
    }
    let report = glrmask::__private::parser_backend_report(constraint);
    assert!(report["static_boundary_shards"].as_u64().unwrap_or(0) > 0, "{report}");
    check(&report);
}

#[test]
fn strict_static_crossings_preserve_delayed_lexemes_sparse_ids_and_aliases() {
    const CHILD: &str = "GLRMASK_TEMPLATE_STATIC_EXCLUSIONS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("strict_static_crossings_preserve_delayed_lexemes_sparse_ids_and_aliases")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let (_, mut tokens) = vocabulary();
    tokens.extend([(1000, b"xaby".to_vec()), (1001, b"xaby".to_vec()), (1033, b"xabby".to_vec()),
        (2048, b"abby".to_vec()), (2051, b"aay".to_vec()), (2052, b"xaaay".to_vec()),
        (2060, b"aq".to_vec()), (3000, b"ayq".to_vec())]);
    let vocab = Vocab::new(tokens.clone());
    let cases: &[(&str, &[&[u8]])] = &[
        (r#"start root; t A = "a" | "ab"; nt root = A "b"?;"#,
            &[b"xay", b"xaby", b"xabby"]),
        (r#"start root; t A = "a" | "aa"; nt root = A "a"?;"#,
            &[b"xay", b"xaay", b"xaaay"]),
    ];
    let options = || MaskMode::ParserDwa.build_options().parser_backend(ParserBackend::TemplateDfa);
    for (source, words) in cases {
        let child = Grammar::from_glrm(source).unwrap().compile_with_build_options(&vocab, options()).unwrap();
        let bound = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap().bind_snapshot("child", &child).unwrap();
        let candidate = bound.link_with(options()).unwrap();
        let loaded = Constraint::load(candidate.save(), &vocab).unwrap();
        let external = Constraint::load(candidate.save(), &vocab).unwrap();
        for candidate in [&candidate, &loaded, &external] {
            assert_static_boundaries(candidate);
            assert_language(candidate, &tokens, words);
            assert!(!candidate.start().mask().get(999 / 32).is_some_and(|word| word & (1 << (999 % 32)) != 0));
        }
    }
}

#[test]
fn strict_static_table_free_nested_composition_uses_no_dynamic_boundary() {
    const CHILD: &str = "GLRMASK_TEMPLATE_STATIC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("strict_static_table_free_nested_composition_uses_no_dynamic_boundary")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let (vocab, tokens) = vocabulary();
    let build = |optimization: MaskMode| optimization.build_options().parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a" | "b""#).unwrap().compile_with_build_options(&vocab, build(MaskMode::ParserDwa)).unwrap();
    // The nested component starts with a dynamic boundary. The outer static
    // construction must materialize that boundary too, not hide a fallback.
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = "p" leaf "q";"#).unwrap()
        .prepare(&vocab).unwrap().bind_snapshot("leaf", &leaf).unwrap().link_with(build(MaskMode::C)).unwrap();
    let middle = Constraint::load(middle.save(), &vocab).unwrap();
    let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#).unwrap()
        .prepare(&vocab).unwrap().bind_snapshot("middle", &middle).unwrap().link_with(build(MaskMode::ParserDwa)).unwrap();
    let words: &[&[u8]] = &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"];
    let loaded = Constraint::load(outer.save(), &vocab).unwrap();
    let external = Constraint::load(outer.save(), &vocab).unwrap();
    for candidate in [&outer, &loaded, &external] {
        assert_static_boundaries(candidate);
        assert_language(candidate, &tokens, words);
    }
    // The user's reusable dynamic child is immutable and remains reusable.
    let original = glrmask::__private::parser_backend_report(&middle);
    assert!(original["dynamic_boundary_shards"].as_u64().unwrap_or(0) > 0);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow (~24s); run with --features slow-tests"
)]
fn strict_static_nullable_controls_have_no_unfolding_depth_limit() {
    const CHILD: &str = "GLRMASK_TEMPLATE_NULLABLE_STATIC_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact").arg("strict_static_nullable_controls_have_no_unfolding_depth_limit")
            .arg("--nocapture").arg("--test-threads=1")
            .env(CHILD, "1").env("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", "1")
            .output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        return;
    }
    let (vocab, tokens) = vocabulary();
    let options = || MaskMode::ParserDwa.build_options().parser_backend(ParserBackend::TemplateDfa);
    let child = Grammar::from_ebnf(r#"start ::= "a"?"#).unwrap().compile_with_build_options(&vocab, options()).unwrap();
    // A single gap can require 128 CALL/RETURN events despite link depth 1.
    let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = "x" child{64} "y";"#).unwrap()
        .prepare(&vocab).unwrap();
    let c = parent.bind_snapshot("child", &child).unwrap().link_with(options()).unwrap();
    let words = (0..=64).map(|count| format!("x{}y", "a".repeat(count)).into_bytes()).collect::<Vec<_>>();
    let words = words.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let loaded = Constraint::load(c.save(), &vocab).unwrap();
    let external = Constraint::load(c.save(), &vocab).unwrap();
    for c in [&c, &loaded, &external] { assert_static_boundaries(c); assert_language(c, &tokens, &words); }

    // Nullable repetition has no finite control-word enumeration bound.
    // Its language is x a* y; check against a direct byte-language predicate.
    let parent = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = "x" child* "y";"#).unwrap()
        .prepare(&vocab).unwrap();
    let c = parent.bind_snapshot("child", &child).unwrap().link_with(options()).unwrap();
    let loaded = Constraint::load(c.save(), &vocab).unwrap();
    fn viable(word: &[u8]) -> bool {
        if word.is_empty() { return true; }
        if word[0] != b'x' { return false; }
        let tail = word[1..].strip_suffix(b"y").unwrap_or(&word[1..]);
        tail.iter().all(|&byte| byte == b'a')
    }
    for c in [&c, &loaded] {
        assert_static_boundaries(c);
        for count in [0usize, 1, 2, 7, 31, 65, 128] {
            let prefix = format!("x{}", "a".repeat(count)).into_bytes();
            let mut state = c.start(); assert!(state.commit_bytes(&prefix)); assert!(!state.is_accepting());
            let mask = state.mask();
            for (id, bytes) in &tokens {
                let mut candidate = prefix.clone(); candidate.extend(bytes);
                let actual = mask.get(*id as usize / 32).is_some_and(|word| word & (1 << (*id % 32)) != 0);
                assert_eq!(actual, viable(&candidate), "prefix_len={} token={bytes:?}", prefix.len());
            }
            assert!(state.commit_bytes(b"y")); assert!(state.is_accepting());
        }
    }
}

#[test]
fn nullable_nested_table_free_children_keep_every_repetition_count() {
    let (vocab, tokens) = vocabulary();
    let options = || glrmask::test_modes::MaskMode::C.build_options().parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a"?"#).unwrap().compile_with_build_options(&vocab, options()).unwrap();
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = leaf leaf;"#).unwrap()
        .prepare(&vocab).unwrap().bind_snapshot("leaf", &leaf).unwrap().link_with(options()).unwrap();
    for middle in [&middle, &Constraint::load(middle.save(), &vocab).unwrap()] {
        let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#).unwrap()
            .prepare(&vocab).unwrap().bind_snapshot("middle", middle).unwrap().link_with(options()).unwrap();
        let words: &[&[u8]] = &[b"xy", b"xay", b"xaay", b"xaaay", b"xaaaay"];
        assert_language(&outer, &tokens, words);
        assert_language(&Constraint::load(outer.save(), &vocab).unwrap(), &tokens, words);
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow (~80s); run with --features slow-tests"
)]
fn nested_nullable_return_keeps_the_empty_child_crossing_mask() {
    let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    for bytes in ["X[a]!", "X[]!", "X[ab]!", "[a]", "a]!", "[]!", "X!",
        "Xa!", "a!", "X[", "a", "a", "dead-branch"] {
        tokens.push((tokens.len() as u32, bytes.as_bytes().to_vec()));
    }
    let vocab = Vocab::new(tokens.clone());
    let words: &[&[u8]] = &[b"X[]!", b"X[a]!"];
    let prefixes = words.iter().flat_map(|word| (0..=word.len()).map(|end| &word[..end]))
        .collect::<BTreeSet<_>>();
    let options = |mode: MaskMode| mode.build_options().parser_backend(ParserBackend::TemplateDfa);
    let oracle = Grammar::from_glrm(r#"glrm 1; start root; nt root = "X" "[" "a"? "]" "!";"#).unwrap()
        .compile_with_build_options(&vocab, options(MaskMode::ParserDwa)).unwrap();
    for leaf_mode in [MaskMode::C, MaskMode::ParserDwa] {
        let leaf = Grammar::from_glrm(r#"glrm 1; start value; nt value = "a"?;"#).unwrap()
            .compile_with_build_options(&vocab, options(leaf_mode)).unwrap();
        for middle_mode in [MaskMode::C, MaskMode::ParserDwa] {
            let middle = Grammar::from_glrm(r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#).unwrap()
                .prepare(&vocab).unwrap().bind_snapshot("leaf", &leaf).unwrap()
                .link_with(options(middle_mode)).unwrap();
            let reloaded_middle = Constraint::load(middle.save(), &vocab).unwrap();
            for middle in [&middle, &reloaded_middle] {
                for outer_mode in [MaskMode::C, MaskMode::ParserDwa] {
                    let linked = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "X" middle "!";"#).unwrap()
                        .prepare(&vocab).unwrap().bind_snapshot("middle", middle).unwrap()
                        .link_with(options(outer_mode)).unwrap();
                    let loaded = Constraint::load(linked.save(), &vocab).unwrap();
                    let external = Constraint::load(linked.save(), &vocab).unwrap();
                    if outer_mode == MaskMode::ParserDwa { assert_static_boundaries(&linked); }
                    for constraint in [&linked, &loaded, &external] {
                        for &prefix in &prefixes {
                            let mut reference = oracle.start(); assert!(reference.commit_bytes(prefix));
                            let expected = reference.mask();
                            let mut state = constraint.start(); assert!(state.commit_bytes(prefix));
                            assert_eq!(state.mask(), expected,
                                "leaf={leaf_mode:?} middle={middle_mode:?} outer={outer_mode:?} prefix={prefix:?}");
                            assert_eq!(state.is_accepting(), words.contains(&prefix));
                            for (id, bytes) in &tokens {
                                let mut word = prefix.to_vec(); word.extend(bytes);
                                let viable = words.iter().any(|candidate| candidate.starts_with(&word));
                                let bit = expected[*id as usize / 32] & (1 << (*id % 32)) != 0;
                                assert_eq!(bit, viable, "independent finite language: prefix={prefix:?} token={bytes:?}");
                                let mut endpoint = state.clone();
                                assert_eq!(endpoint.commit_token(*id), viable,
                                    "mask/commit endpoint: prefix={prefix:?} token={bytes:?}");
                                if viable { assert_eq!(endpoint.is_accepting(), words.contains(&word.as_slice())); }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn scoped_ignores_match_the_independent_lr_composition() {
    let (vocab, _) = vocabulary();
    let parent = Grammar::from_glrm(r#"glrm 1; start root; ignore WS; t WS = "~"+; extern grammar child; nt root = "x" child "y";"#).unwrap()
        .prepare(&vocab).unwrap();
    let grammar = Grammar::from_glrm(r#"start root; ignore WS; t WS = "_"+; nt root = "a" "b"?;"#).unwrap();
    let lr_child = grammar.compile(&vocab).unwrap();
    let lr = parent.bind_snapshot("child", &lr_child).unwrap().link_with(glrmask::test_modes::MaskMode::C.build_options()).unwrap();
    let child = grammar.compile_with_build_options(&vocab, BuildOptions::default().parser_backend(ParserBackend::TemplateDfa)).unwrap();
    let candidate = parent.bind_snapshot("child", &child).unwrap().link_with(
        MaskMode::C.build_options().parser_backend(ParserBackend::TemplateDfa)).unwrap();
    let loaded = Constraint::load(candidate.save(), &vocab).unwrap();
    let external = Constraint::load(candidate.save(), &vocab).unwrap();
    let mut prefixes = vec![vec![]]; let mut layer = vec![vec![]];
    for _ in 0..4 {
        let mut next = Vec::new();
        for word in layer { for byte in b"xayb~_" { let mut extended = word.clone(); extended.push(*byte); next.push(extended); } }
        prefixes.extend(next.iter().cloned()); layer = next;
    }
    prefixes.extend([b"~x_a_by~".to_vec(), b"x_a~by".to_vec(), b"x~a_by".to_vec(), b"x__a__b__y".to_vec()]);
    for prefix in prefixes {
        let mut reference = lr.start(); let valid = reference.commit_bytes(&prefix);
        for constraint in [&candidate, &loaded, &external] {
            let mut state = constraint.start();
            assert_eq!(state.commit_bytes(&prefix), valid, "ignore prefix {prefix:?}");
            if valid {
                assert_eq!(state.is_accepting(), reference.is_accepting(), "ignore completion {prefix:?}");
                assert_eq!(state.mask(), reference.mask(), "ignore mask {prefix:?}");
            }
        }
    }
}

#[test]
fn table_free_links_preserve_exact_empty_tokens_and_root_end_policy() {
    let (_, mut tokens) = vocabulary();
    tokens.extend([(600, b"special".to_vec()), (601, vec![]), (602, b"special".to_vec()), (700, vec![])]);
    let vocab = Vocab::new(tokens);
    let child = Grammar::from_glrm(r#"extern token SPECIAL extern token EMPTY start root nt root = SPECIAL "a" | EMPTY "b""#).unwrap()
        .bind_snapshot("SPECIAL",TokenSet::new([600])).unwrap()
        .bind_snapshot("EMPTY",TokenSet::new([601])).unwrap()
        .compile_with_build_options(&vocab, BuildOptions::default().parser_backend(ParserBackend::TemplateDfa)).unwrap();
    let parent = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap();
    let result = parent.bind_snapshot("child", &child).unwrap().link_with(
        MaskMode::C.build_options().parser_backend(ParserBackend::TemplateDfa).end_tokens([700])).unwrap();
    let loaded = Constraint::load(result.save(), &vocab).unwrap();
    for c in [&result, &loaded] {
        for (special, letter) in [(600, b'a'), (601, b'b')] {
            let mut state = c.start(); assert!(state.commit_bytes(b"x"));
            let allowed = |mask: &[u32], token: u32| mask[token as usize / 32] & (1 << (token % 32)) != 0;
            let mask = state.mask(); assert!(allowed(&mask, special)); assert!(!allowed(&mask, 602)); assert!(!allowed(&mask, 700));
            assert!(state.commit_token(special)); assert!(state.commit_bytes(&[letter, b'y']));
            assert!(allowed(&state.mask(), 700)); assert!(state.commit_token(700));
            
            assert!(state.mask().iter().all(|&w| w == 0));
            assert!(!state.commit_token(special));
        }
    }
}

#[test]
fn already_table_free_children_link_without_reconstructing_tables() {
    let (vocab, tokens) = vocabulary();
    for child_mode in [MaskMode::C, MaskMode::ParserDwa] {
        for nullable in [false, true] {
            let source = if nullable { r#"start ::= "a"?"# } else { r#"start ::= "a" | "b" | "a" "b""# };
            let child = Grammar::from_ebnf(source).unwrap().compile_with_build_options(&vocab,
                child_mode.build_options().parser_backend(ParserBackend::TemplateDfa)).unwrap();
            let loaded = Constraint::load(child.save(), &vocab).unwrap();
            for child in [&child, &loaded] {
                assert_eq!(glrmask::__private::parser_backend_report(child)["finite_embedding"], true);
                let bound = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap().bind_snapshot("child", child).unwrap();
                for mode in MaskMode::ALL {
                    let result = bound.link_with(mode.build_options()
                        .parser_backend(ParserBackend::TemplateDfa)).unwrap();
                    let words: &[&[u8]] = if nullable { &[b"xy", b"xay"] } else { &[b"xay", b"xby", b"xaby"] };
                    assert_language(&result, &tokens, words);
                    assert_language(&Constraint::load(result.save(), &vocab).unwrap(), &tokens, words);
                }
            }
        }
    }
}

#[test]
fn table_free_composition_can_itself_be_reused_as_a_child() {
    let (vocab, tokens) = vocabulary();
    let options = || glrmask::test_modes::MaskMode::C.build_options().parser_backend(ParserBackend::TemplateDfa);
    let leaf = Grammar::from_ebnf(r#"start ::= "a" | "b""#).unwrap().compile_with_build_options(&vocab, options()).unwrap();
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = "p" leaf "q";"#).unwrap()
        .prepare(&vocab).unwrap().bind_snapshot("leaf", &leaf).unwrap().link_with(options()).unwrap();
    for middle in [&middle, &Constraint::load(middle.save(), &vocab).unwrap()] {
        let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#).unwrap()
            .prepare(&vocab).unwrap().bind_snapshot("middle", middle).unwrap().link_with(options()).unwrap();
        let words: &[&[u8]] = &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"];
        assert_language(&outer, &tokens, words);
        assert_language(&Constraint::load(outer.save(), &vocab).unwrap(), &tokens, words);
    }
}

fn vocabulary() -> (Vocab, Vec<(u32, Vec<u8>)>) {
    let mut tokens = (0..128).map(|id| (id, vec![id as u8])).collect::<Vec<_>>();
    for (index, token) in ["xay", "xby", "xaby", "xa", "ay", "ab", "by", "xy", "yx", " ",
        "xp", "paq", "aqy", "xpaqy", "xpaqpaqy", "xpaqpbqy", "qy", "pq" ].iter().enumerate() {
        tokens.push((256 + index as u32, token.as_bytes().to_vec()));
    }
    (Vocab::new(tokens.clone()), tokens)
}

#[test]
fn nested_repeated_child_retains_scope_across_token_boundaries_and_reload() {
    let (vocab, tokens) = vocabulary();
    let leaf = Grammar::from_ebnf(r#"start ::= "a" | "b""#).unwrap().compile(&vocab).unwrap();
    let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = "p" leaf "q";"#).unwrap()
        .prepare(&vocab).unwrap().bind_snapshot("leaf", &leaf).unwrap().link().unwrap();
    let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#).unwrap()
        .prepare(&vocab).unwrap().bind_snapshot("middle", &middle).unwrap();
    for optimization in [MaskMode::C, MaskMode::ParserDwa] {
        let candidate = outer.link_with(optimization.build_options()
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"]);
        let loaded = Constraint::load(candidate.save(), &vocab).unwrap();
        assert_language(&loaded, &tokens, &[b"xpaqpaqy", b"xpaqpbqy", b"xpbqpaqy", b"xpbqpbqy"]);
    }
}

fn assert_language(constraint: &Constraint, tokens: &[(u32, Vec<u8>)], words: &[&[u8]]) {
    assert_eq!(constraint.parser_backend(), ParserBackend::TemplateDfa);
    let report = glrmask::__private::parser_backend_report(constraint);
    assert_eq!(report["lr_table_present"], false);
    let mut prefixes = BTreeSet::new();
    for word in words {
        for length in 0..=word.len() { prefixes.insert(word[..length].to_vec()); }
    }
    for prefix in prefixes {
        let mut state = constraint.start();
        assert!(state.commit_bytes(&prefix));
        assert_eq!(state.is_accepting(), words.contains(&prefix.as_slice()), "completion {prefix:?}");
        let mask = state.mask();
        for (id, bytes) in tokens {
            let mut candidate = prefix.clone(); candidate.extend(bytes);
            let expected = words.iter().any(|word| word.starts_with(&candidate));
            let actual = mask.get(*id as usize / 32).is_some_and(|word| word & (1 << (*id % 32)) != 0);
            assert_eq!(actual, expected, "prefix={prefix:?}, token={bytes:?}, id={id}, report={report}");
            if actual {
                let mut replay = constraint.start();
                assert!(replay.commit_bytes(&prefix));
                assert!(replay.commit_token(*id));
                assert_eq!(replay.is_accepting(), words.contains(&candidate.as_slice()), "commit {candidate:?}");
            }
        }
    }
}

#[test]
fn compiled_child_composition_uses_templates_for_advance_and_masks() {
    let (vocab, tokens) = vocabulary();
    let child = Grammar::from_ebnf(r#"start ::= "a" | "b" | "a" "b""#).unwrap().compile(&vocab).unwrap();
    let bound = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap().bind_snapshot("child", &child).unwrap();
    for optimization in MaskMode::ALL {
        let candidate = bound.link_with(optimization.build_options()
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xay", b"xby", b"xaby"]);
        let saved = candidate.save();
        let loaded = Constraint::load(&saved, &vocab).unwrap();
        assert_language(&loaded, &tokens, &[b"xay", b"xby", b"xaby"]);
        assert_eq!(saved, loaded.save());
        assert_eq!(child.parser_backend(), ParserBackend::TemplateDfa, "link must not mutate a shared child");
    }
}

#[test]
fn source_bound_composition_uses_template_backend() {
    let (vocab, tokens) = vocabulary();
    let child = Grammar::from_ebnf(r#"start ::= "a" | "b" | "a" "b""#).unwrap();
    let source = Grammar::from_glrm(HOST).unwrap().bind_snapshot("child", &child).unwrap();
    for optimization in [MaskMode::B, MaskMode::C, MaskMode::ParserDwa] {
        let candidate = source.compile_with_build_options(&vocab, optimization.build_options()
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xay", b"xby", b"xaby"]);
        let loaded = Constraint::load(candidate.save(), &vocab).unwrap();
        assert_language(&loaded, &tokens, &[b"xay", b"xby", b"xaby"]);
    }
}

#[test]
fn nullable_child_controls_preserve_empty_body_and_parent_suffix() {
    let (vocab, tokens) = vocabulary();
    let child = Grammar::from_ebnf(r#"start ::= "a"?"#).unwrap().compile(&vocab).unwrap();
    let bound = Grammar::from_glrm(HOST).unwrap().prepare(&vocab).unwrap().bind_snapshot("child", &child).unwrap();
    for optimization in [MaskMode::C, MaskMode::ParserDwa] {
        let candidate = bound.link_with(optimization.build_options()
            .parser_backend(ParserBackend::TemplateDfa)).unwrap();
        assert_language(&candidate, &tokens, &[b"xy", b"xay"]);
        let loaded = Constraint::load(candidate.save(), &vocab).unwrap();
        assert_language(&loaded, &tokens, &[b"xy", b"xay"]);
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow (~50s); run with --features slow-tests"
)]
fn nested_artifact_loads_preserve_exact_language_on_default_rayon_stacks() {
    use rayon::prelude::*;

    let (vocab, tokens) = vocabulary();
    let words: &[&[u8]] = &[
        b"xxpaqpaqyy",
        b"xxpaqpbqyy",
        b"xxpbqpaqyy",
        b"xxpbqpbqyy",
    ];

    let build = |optimization: MaskMode| {
        optimization.build_options()
            .parser_backend(ParserBackend::TemplateDfa)
    };

    for optimization in [MaskMode::C, MaskMode::ParserDwa] {
        let leaf = Grammar::from_ebnf(r#"start ::= "a" | "b""#).unwrap()
            .compile_with_build_options(&vocab, build(optimization))
            .unwrap();
        let middle = Grammar::from_glrm(r#"glrm 1; start mid; extern grammar leaf; nt mid = "p" leaf "q";"#).unwrap()
            .prepare(&vocab)
            .unwrap()
            .bind_snapshot("leaf", &leaf)
            .unwrap()
            .link_with(build(optimization))
            .unwrap();
        let outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle middle "y";"#).unwrap()
            .prepare(&vocab)
            .unwrap()
            .bind_snapshot("middle", &middle)
            .unwrap()
            .link_with(build(optimization))
            .unwrap();
        let topwrap = Grammar::from_glrm(HOST).unwrap()
            .prepare(&vocab)
            .unwrap()
            .bind_snapshot("child", &outer)
            .unwrap()
            .link_with(build(optimization))
            .unwrap();

        assert_language(&topwrap, &tokens, words);
        let saved = topwrap.save();

        for workers in [2, 4] {
            eprintln!("[template_loader_stress] optimization={optimization:?} workers={workers}");
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|id| format!("native-loader-stress-{id}"))
                .build()
                .unwrap();

            for _round in 0..8 {
                pool.install(|| {
                    (0..16usize).into_par_iter().for_each(|_| {
                        let loaded = Constraint::load(&saved, &vocab).unwrap();
                        assert_language(&loaded, &tokens, words);
                        assert_eq!(saved, loaded.save());
                    });
                });
            }
        }
    }
}
