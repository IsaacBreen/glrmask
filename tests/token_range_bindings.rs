use glrmask::{BuildOptions, Constraint, Grammar, Optimization, PreparedGrammar, Vocab};

const ONE: &str = "glrm 1; start start; extern token T; nt start = T;";
const MODES: [Optimization; 4] = [Optimization::Auto, Optimization::FastBuild, Optimization::Balanced, Optimization::FastRuntime];

fn allowed(mask: &[u32], token: u32) -> bool {
    mask.get(token as usize / 32).is_some_and(|word| word & (1 << (token % 32)) != 0)
}

fn assert_mask(constraint: &Constraint, prefix: &[u32], domain: u32, expected: impl Fn(u32) -> bool) {
    let mut state = constraint.start();
    for &token in prefix { state.commit_token(token).unwrap(); }
    let mask = state.mask();
    for token in 0..domain {
        assert_eq!(allowed(&mask, token), expected(token), "prefix={prefix:?}, token={token}");
    }
}

fn mixed_vocab() -> Vocab {
    // Byte aliases deliberately straddle interval boundaries. Empty-byte and
    // exact-only IDs are included, and 40..48 is an undefined hole.
    Vocab::new_with_exact_token_ids(
        (0..40).map(|id| (id, if id % 3 == 0 { Vec::new() } else { b"x".to_vec() })).collect(),
        48..80,
    )
}

#[test]
fn exact_range_bindings_cover_all_members_without_byte_alias_leaks() {
    let vocab = mixed_vocab();
    let tokens = vocab.token_ranges([5..=18, 31..=33, 48..=63]).unwrap();
    assert_eq!(tokens.ranges().collect::<Vec<_>>(), [5..=18, 31..=33, 48..=63]);
    let accepts = |id| (5..=18).contains(&id) || (31..=33).contains(&id) || (48..=63).contains(&id);
    for mode in MODES {
        let options = BuildOptions::default().optimization(mode).end_tokens([95]);
        let constraint = Grammar::from_glrm(ONE).bind("T", tokens.clone()).unwrap()
            .compile_with(&vocab, options).unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        assert_mask(&constraint, &[], 96, accepts);
        assert!(constraint.start().forced().is_empty(), "a multi-token set is not forced");
        for id in [5, 6, 18, 31, 33, 48, 63] {
            let mut state = constraint.start();
            state.commit_token(id).unwrap();
            assert!(state.is_accepting(), "{mode:?}, {id}");
            let mask = state.mask();
            assert!(allowed(&mask, 95));
            assert_eq!(mask.iter().map(|w| w.count_ones()).sum::<u32>(), 1);
        }
        let loaded = Constraint::load(&constraint.save()).unwrap();
        assert_mask(&loaded, &[], 96, accepts);
        let external = constraint.save_without_vocab().unwrap();
        let external = Constraint::load_with_vocab(&external, &vocab).unwrap();
        assert_mask(&external, &[], 96, accepts);
    }
}

#[test]
fn overlapping_ranges_query_each_terminal_not_a_representative_token() {
    let vocab = Vocab::new_with_exact_token_ids(Vec::<(u32, Vec<u8>)>::new(), 0..100);
    // At the start only A is live. After A only B is live. Their overlap must
    // not make either one's non-overlapping tail spuriously allowed.
    let source = "glrm 1; start start; extern token A; extern token B; nt start = A B;";
    for mode in MODES {
        let constraint = Grammar::from_glrm(source)
            .bind("A", vocab.token_ranges([10..=30]).unwrap()).unwrap()
            .bind("B", vocab.token_ranges([20..=40]).unwrap()).unwrap()
            .compile_with(&vocab, BuildOptions::default().optimization(mode)).unwrap();
        assert_mask(&constraint, &[], 100, |id| (10..=30).contains(&id));
        for first in [10, 20, 25, 30] {
            assert_mask(&constraint, &[first], 100, |id| (20..=40).contains(&id));
            let mut state = constraint.start();
            state.commit_token(first).unwrap();
            state.commit_token(40).unwrap();
            assert!(state.is_accepting());
        }
    }
}

#[test]
fn repeated_ranges_and_compiled_open_slots_roundtrip() {
    let vocab = Vocab::new_with_exact_token_ids(vec![(0, b"[".to_vec()), (1, b"]".to_vec())], 2..128);
    let child_source = "glrm 1; start start; extern token T; nt start = T T;";
    let open = Grammar::from_glrm(child_source).prepare(&vocab).unwrap();
    let bound = open.bind("T", vocab.token_ranges([5..=15, 100..=120]).unwrap()).unwrap();
    let serialized = bound.save();
    let loaded = PreparedGrammar::load(&serialized).unwrap();
    for mode in MODES {
        let options = BuildOptions::default().optimization(mode);
        let child = loaded.link_with(options.clone()).unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        assert_mask(&child, &[], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        assert_mask(&child, &[15], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        // Reuse the same closed prepared body without linking it prematurely.
        // Each root link chooses its requested optimization independently.
        let prepared_child = loaded.clone();
        let outer = Grammar::from_glrm("glrm 1; start start; extern grammar C; nt start = \"[\" C \"]\";")
            .prepare(&vocab).unwrap().bind("C", &prepared_child).unwrap()
            .link_with(options).unwrap();
        assert_mask(&outer, &[0], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        let mut state = outer.start();
        for id in [0, 120, 5, 1] { state.commit_token(id).unwrap(); }
        assert!(state.is_accepting());
        let outer_loaded = Constraint::load(&outer.save()).unwrap();
        assert_mask(&outer_loaded, &[0, 120], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
    }
    // Binding is immutable: the original description is still open.
    assert!(open.link().is_err());
}

#[test]
fn range_construction_normalizes_and_rejects_missing_ids() {
    let vocab = mixed_vocab();
    let ranges = vocab.token_ranges([12..=20, 5..=12, 21..=25, 6..=7]).unwrap();
    assert_eq!(ranges.ranges().collect::<Vec<_>>(), [5..=25]);
    assert_eq!(ranges.len(), 21);
    assert_eq!(ranges.ids(), &(5..=25).collect::<Vec<_>>());
    assert!(vocab.token_ranges([]).unwrap().is_empty());
    assert!(vocab.tokens([]).unwrap().is_empty());
    assert!(vocab.token_ranges([std::ops::RangeInclusive::new(20, 10)]).is_err());
    assert!(vocab.token_ranges([39..=48]).is_err());
    assert!(vocab.token_ranges([79..=80]).is_err());
    assert!(vocab.tokens([5, 5]).is_err(), "existing duplicate-list validation is preserved");
    let different = Vocab::new_with_exact_token_ids(vec![(0, b"different".to_vec())], 1..80);
    assert!(Grammar::from_glrm(ONE).bind("T", ranges).unwrap().compile(&different).is_err());
}

#[test]
fn full_128k_and_nearly_full_vocabularies_compile_and_mask_compactly() {
    const N: u32 = 131_072;
    let vocab = Vocab::new_with_exact_token_ids(Vec::<(u32, Vec<u8>)>::new(), 0..N);
    for excluded in [None, Some(65_535)] {
        let ranges = match excluded {
            None => vec![0..=N - 1],
            Some(id) => vec![0..=id - 1, id + 1..=N - 1],
        };
        let tokens = vocab.token_ranges(ranges).unwrap();
        assert_eq!(tokens.ranges().len(), if excluded.is_some() { 2 } else { 1 });
        assert_eq!(tokens.len(), (N - u32::from(excluded.is_some())) as usize);
        for mode in MODES {
            let constraint = Grammar::from_glrm(ONE).bind("T", tokens.clone()).unwrap()
                .compile_with(&vocab, BuildOptions::default().optimization(mode))
                .unwrap_or_else(|e| panic!("{mode:?}, excluded={excluded:?}: {e}"));
            let mask = constraint.start().mask();
            assert_eq!(mask.iter().map(|w| w.count_ones()).sum::<u32>(), N - u32::from(excluded.is_some()));
            for id in [0, 31, 32, 65_534, 65_535, 65_536, N - 1] {
                assert_eq!(allowed(&mask, id), excluded != Some(id));
                if excluded != Some(id) {
                    let mut state = constraint.start();
                    state.commit_token(id).unwrap();
                    assert!(state.is_accepting());
                }
            }
        }
    }
}

#[test]
fn range_grammar_structure_depends_on_intervals_not_cardinality() {
    use glrmask_grammar::__private::grammar::{ast, glrm};
    for end in [31, 131_071] {
        let named = glrm::from_glrm_with_external_terminal_ranges(
            ONE, &[("T", &[(0, end)])],
        ).unwrap();
        let flat = ast::lower(&named).unwrap();
        assert!(flat.terminals.len() <= 2, "one interval must remain one exact-token terminal");
        assert!(flat.rules.len() <= 4);
    }
    let named = glrm::from_glrm_with_external_terminal_ranges(
        ONE, &[("T", &[(0, 65_534), (65_536, 131_071)])],
    ).unwrap();
    let flat = ast::lower(&named).unwrap();
    assert!(flat.terminals.len() <= 3);
    assert!(flat.rules.len() <= 6);
}
#[test]
fn exact_ranges_interact_with_byte_alternatives_without_cross_boundary_aliases() {
    let mut entries = vec![
        (0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"bb".to_vec()),
        (3, b"ab".to_vec()), (4, b"abb".to_vec()), (5, Vec::new()),
        (6, b"x".to_vec()), (7, b"b".to_vec()),
    ];
    entries.extend((8..20).map(|id| (id, b"x".to_vec())));
    let vocab = Vocab::new(entries);
    let source = r#"glrm 1; start start; extern token A; extern token B;
        nt start = ("a" A "b") | ("a" B "bb");"#;
    for mode in MODES {
        let constraint = Grammar::from_glrm(source)
            .bind("A", vocab.token_ranges([8..=12]).unwrap()).unwrap()
            .bind("B", vocab.token_ranges([10..=14]).unwrap()).unwrap()
            .compile_with(&vocab, BuildOptions::default().optimization(mode)).unwrap();
        assert_mask(&constraint, &[], 20, |id| id == 0);
        assert_mask(&constraint, &[0], 20, |id| (8..=14).contains(&id));
        for exact in 8..=14 {
            assert_mask(&constraint, &[0, exact], 20,
                |id| id == 1 || id == 7 || (exact >= 10 && id == 2));
            let mut state = constraint.start();
            state.commit_token(0).unwrap();
            let mut independent = state.clone();
            let before = independent.mask();
            state.commit_token(exact).unwrap();
            assert_eq!(independent.mask(), before);
            // Byte aliases outside the set are not accepted as exact IDs.
            assert!(independent.commit_token(6).is_err());
            state.commit_token(1).unwrap();
            assert_eq!(state.is_accepting(), exact <= 12);
            if exact >= 10 {
                state.commit_token(1).unwrap();
                assert!(state.is_accepting());
            }
        }
    }
}
#[test]
fn ranges_preserve_forcing_and_reserved_end_token_policy() {
    let vocab = Vocab::new_with_exact_token_ids(Vec::new(), 0..16);
    for mode in MODES {
        let singleton = Grammar::from_glrm(ONE)
            .bind("T", vocab.token_ranges([5..=5]).unwrap()).unwrap()
            .compile_with(&vocab, BuildOptions::default().optimization(mode)).unwrap();
        assert_eq!(singleton.start().forced(), vec![5]);
        let repeated = Grammar::from_glrm("glrm 1; start start; extern token T; nt start = T T;")
            .bind("T", vocab.token_ranges([0..=15]).unwrap()).unwrap()
            .compile_with(&vocab, BuildOptions::default().optimization(mode).end_tokens([8])).unwrap();
        assert_mask(&repeated, &[], 16, |id| id != 8);
        assert_mask(&repeated, &[7], 16, |id| id != 8);
        assert_mask(&repeated, &[7, 9], 16, |id| id == 8);
        let mut early = repeated.start();
        early.commit_token(8).unwrap();
        assert!(early.is_rejected());
        assert!(!early.is_accepting());
        assert!(early.mask().iter().all(|&word| word == 0));
        let mut state = repeated.start();
        for id in [7, 9, 8] { state.commit_token(id).unwrap(); }
        assert!(state.is_accepting());
        assert!(state.mask().iter().all(|&word| word == 0));
    }
}

#[test]
fn neutral_scalar_module_formats_are_not_misread_as_range_artifacts() {
    let vocab = Vocab::new_with_exact_token_ids(Vec::new(), 0..128);
    let module = Grammar::from_glrm(ONE).prepare(&vocab).unwrap()
        .bind("T", vocab.token_ranges([0..=127]).unwrap()).unwrap();
    let bytes = module.save();
    for version in [b"GLRMOD03", b"GLRMOD04", b"GLRMOD05", b"GLRMOD06"] {
        let mut old = bytes.clone();
        old[..8].copy_from_slice(version);
        assert!(PreparedGrammar::load(&old).is_err());
    }
}

#[test]
fn ambiguous_exact_and_byte_choices_do_not_admit_a_third_token() {
    let vocab = Vocab::new(vec![
        (0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec()),
        (3, b"aa".to_vec()), (4, b"ba".to_vec()), (5, b"x".to_vec()),
        (6, b"a".to_vec()), (7, Vec::new()), (8, b"b".to_vec()),
        (9, b"c".to_vec()), (10, b"bb".to_vec()), (11, b"x".to_vec()),
    ]);
    let source = r#"glrm 1; start start; extern token A; extern token B; nt start = (A | B) (A | "a");"#;
    for mode in MODES {
        for (a, b) in [(0..=0, 1..=1), (0..=5, 3..=9), (0..=11, 0..=10)] {
            let grammar = Grammar::from_glrm(source)
                .bind("A", vocab.token_ranges([a]).unwrap()).unwrap()
                .bind("B", vocab.token_ranges([b]).unwrap()).unwrap();
            let options = BuildOptions::default().optimization(mode).end_tokens([15]);
            let constraint = grammar.compile_with(&vocab, options).unwrap();
            for constraint in [constraint.clone(), Constraint::load(&constraint.save()).unwrap()] {
                assert_mask(&constraint, &[0, 0], 16, |id| id == 15);
                let mut state = constraint.start();
                state.commit_token(0).unwrap();
                state.commit_token(0).unwrap();
                assert!(state.is_accepting());
                assert!(state.commit_token(0).is_err(), "{mode:?}");
            }
        }
    }
}

#[test]
fn full_empty_spelled_ranges_stay_exact_across_grammar_positions() {
    let n = 32_768u32;
    let vocab = Vocab::new((0..n).map(|id| (id, Vec::new())).collect());
    for mode in MODES {
        let grammar = Grammar::from_glrm("glrm 1; start start; extern token A; extern token B; nt start = A B;")
            .bind("A", vocab.token_ranges([0..=n - 1]).unwrap()).unwrap()
            .bind("B", vocab.token_ranges([0..=n / 2 - 1, n / 2 + 1..=n - 1]).unwrap()).unwrap();
        let original = grammar.compile_with(&vocab,
            BuildOptions::default().optimization(mode).end_tokens([n])).unwrap();
        for constraint in [original.clone(), Constraint::load(&original.save()).unwrap()] {
            assert_mask(&constraint, &[], n + 1, |id| id < n);
            assert_mask(&constraint, &[0], n + 1, |id| id < n && id != n / 2);
            assert_mask(&constraint, &[0, n - 1], n + 1, |id| id == n);
        }
    }
}
