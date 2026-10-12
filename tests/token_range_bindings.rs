use glrmask::{BuildOptions, Constraint, Grammar, Optimization, UnlinkedConstraint, Vocab};

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
    let open = Grammar::from_glrm(child_source).compile_unlinked(&vocab).unwrap();
    let bound = open.bind("T", vocab.token_ranges([5..=15, 100..=120]).unwrap()).unwrap();
    let serialized = bound.save();
    let loaded = UnlinkedConstraint::load(&serialized).unwrap();
    for mode in MODES {
        let options = BuildOptions::default().optimization(mode);
        let child = loaded.link_with(options.clone()).unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        assert_mask(&child, &[], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        assert_mask(&child, &[15], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        // Composition intentionally accepts only native compiled components.
        // Test each final optimization without attaching a retained-LR child.
        let native_child = loaded.link().unwrap();
        let outer = Grammar::from_glrm("glrm 1; start start; extern grammar C; nt start = \"[\" C \"]\";")
            .compile_unlinked(&vocab).unwrap().bind("C", &native_child).unwrap()
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
    assert!(vocab.token_ranges([]).is_err());
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
