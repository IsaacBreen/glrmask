use glrmask::test_modes::MaskMode;
use glrmask::{Constraint, Grammar, PreparedGrammar, TokenSet, Vocab};

const ONE: &str = "glrm 1; start start; extern token T; nt start = T;";
const MODES: [MaskMode; 6] = MaskMode::ALL;

fn allowed(mask: &[u32], token: u32) -> bool {
    mask.get(token as usize / 32).is_some_and(|word| word & (1 << (token % 32)) != 0)
}

fn assert_mask(constraint: &Constraint, prefix: &[u32], domain: u32, expected: impl Fn(u32) -> bool) {
    let mut state = constraint.start();
    for &token in prefix { assert!(state.commit_token(token)); }
    let mask = state.mask();
    for token in 0..domain {
        assert_eq!(allowed(&mask, token), expected(token), "prefix={prefix:?}, token={token}");
    }
}

fn mixed_vocab() -> Vocab {
    // Byte aliases deliberately straddle interval boundaries. Empty-byte and
    // exact-only IDs are included, and 40..48 is an undefined hole.
    Vocab::new((0..40).map(|id| (id, if id % 3 == 0 { Vec::new() } else { b"x".to_vec() })).collect()).with_special_tokens(48..80)
}

#[test]
fn exact_range_bindings_cover_all_members_without_byte_alias_leaks() {
    let vocab = mixed_vocab();
    let tokens = TokenSet::from_ranges([5..=18, 31..=33, 48..=63]);
    assert_eq!(tokens.range_pairs(), &[(5, 18), (31, 33), (48, 63)]);
    let accepts = |id| (5..=18).contains(&id) || (31..=33).contains(&id) || (48..=63).contains(&id);
    for mode in MODES {
        let options = mode.build_options().end_tokens([95]);
        let constraint = Grammar::from_glrm(ONE).unwrap().bind_snapshot("T", tokens.clone()).unwrap()
            .compile_with_build_options(&vocab, options).unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        assert_mask(&constraint, &[], 96, accepts);
        assert!(constraint.start().forced_canonical_reference().is_empty(), "a multi-token set is not forced");
        for id in [5, 6, 18, 31, 33, 48, 63] {
            let mut state = constraint.start();
            assert!(state.commit_token(id));
            // The event completes the grammar; acceptance needs the end token.
            assert!(!state.is_accepting(), "{mode:?}, {id}");
            let mask = state.mask();
            assert!(allowed(&mask, 95));
            assert_eq!(mask.iter().map(|w| w.count_ones()).sum::<u32>(), 1);
            assert!(state.commit_token(95));
            assert!(state.is_accepting(), "{mode:?}, {id}");
            assert!(state.mask().iter().all(|word| *word == 0));
        }
        let loaded = Constraint::load(constraint.save(), &vocab).unwrap();
        assert_mask(&loaded, &[], 96, accepts);
        let external = constraint.save();
        let external = Constraint::load(&external, &vocab).unwrap();
        assert_mask(&external, &[], 96, accepts);
    }
}

#[test]
fn overlapping_ranges_query_each_terminal_not_a_representative_token() {
    let vocab = Vocab::new(Vec::<(u32, Vec<u8>)>::new()).with_special_tokens(0..100);
    // At the start only A is live. After A only B is live. Their overlap must
    // not make either one's non-overlapping tail spuriously allowed.
    let source = "glrm 1; start start; extern token A; extern token B; nt start = A B;";
    for mode in MODES {
        let constraint = Grammar::from_glrm(source).unwrap()
            .bind_snapshot("A", TokenSet::from_ranges([10..=30])).unwrap()
            .bind_snapshot("B", TokenSet::from_ranges([20..=40])).unwrap()
            .compile_with_build_options(&vocab, mode.build_options()).unwrap();
        assert_mask(&constraint, &[], 100, |id| (10..=30).contains(&id));
        for first in [10, 20, 25, 30] {
            assert_mask(&constraint, &[first], 100, |id| (20..=40).contains(&id));
            let mut state = constraint.start();
            assert!(state.commit_token(first));
            assert!(state.commit_token(40));
            assert!(state.is_accepting());
        }
    }
}

#[test]
fn repeated_ranges_and_compiled_open_slots_roundtrip() {
    let vocab = Vocab::new(vec![(0, b"[".to_vec()), (1, b"]".to_vec())]).with_special_tokens(2..128);
    let child_source = "glrm 1; start start; extern token T; nt start = T T;";
    let open = Grammar::from_glrm(child_source).unwrap().prepare(&vocab).unwrap();
    let bound = open.bind_snapshot("T", TokenSet::from_ranges([5..=15, 100..=120])).unwrap();
    let serialized = bound.save();
    let loaded = PreparedGrammar::load(&serialized, &vocab).unwrap();
    for mode in MODES {
        let options = mode.build_options();
        let child = loaded.link_with(options.clone()).unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        assert_mask(&child, &[], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        assert_mask(&child, &[15], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        // Reuse the same closed prepared body without linking it prematurely.
        // Each root link chooses its requested optimization independently.
        let prepared_child = loaded.clone();
        let outer = Grammar::from_glrm("glrm 1; start start; extern grammar C; nt start = \"[\" C \"]\";").unwrap()
            .prepare(&vocab).unwrap().bind_snapshot("C", &prepared_child).unwrap()
            .link_with(options).unwrap();
        assert_mask(&outer, &[0], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
        let mut state = outer.start();
        for id in [0, 120, 5, 1] { assert!(state.commit_token(id)); }
        assert!(state.is_accepting());
        let outer_loaded = Constraint::load(outer.save(), &vocab).unwrap();
        assert_mask(&outer_loaded, &[0, 120], 128, |id| (5..=15).contains(&id) || (100..=120).contains(&id));
    }
    // Binding is immutable: the original description is still open.
    assert!(open.link().is_err());
}

#[test]
fn a_bare_integer_binds_one_token_id() {
    let vocab = Vocab::new(Vec::new()).with_special_tokens(0..8);
    let constraint = Grammar::from_glrm(ONE).unwrap().bind_snapshot("T", 5).unwrap().compile(&vocab).unwrap();
    let mask = constraint.start().mask();
    assert!(allowed(&mask, 5));
    assert_eq!(mask.iter().map(|word| word.count_ones()).sum::<u32>(), 1);
}

#[test]
fn range_construction_normalizes_and_validates_membership_at_compile() {
    let vocab = mixed_vocab();
    let ranges = TokenSet::from_ranges([12..=20, 5..=12, 21..=25, 6..=7]);
    assert_eq!(ranges, TokenSet::from_ranges([5..=25]));
    assert_eq!(
        TokenSet::from_ranges(std::iter::empty::<std::ops::RangeInclusive<u32>>()),
        TokenSet::new([]),
    );
    assert_eq!(TokenSet::from_ranges(std::iter::once(std::ops::RangeInclusive::new(20, 10))).range_pairs(), &[]);
    assert_eq!(TokenSet::from_ranges(std::iter::once(5..5)).range_pairs(), &[]);
    assert_eq!(TokenSet::new([5, 5]), TokenSet::from(5));
    let crossing_hole = Grammar::from_glrm(ONE).unwrap().bind_snapshot("T", TokenSet::from_ranges([39..=48])).unwrap();
    assert!(crossing_hole.compile(&vocab).is_err(), "a range may not span undefined IDs");
    let outside = Grammar::from_glrm(ONE).unwrap().bind_snapshot("T", TokenSet::from_ranges([79..=80])).unwrap();
    assert!(outside.compile(&vocab).is_err(), "an ID outside the vocabulary is rejected");
    let different = Vocab::new(vec![(0, b"different".to_vec())]).with_special_tokens(1..80);
    // A token set is IDs only: different byte spellings do not matter, but every
    // ID must be defined by the target vocabulary.
    Grammar::from_glrm(ONE).unwrap().bind_snapshot("T", ranges).unwrap().compile(&different).unwrap();
    // The same IDs remain valid against any vocabulary that defines them.
    let same_ids = Vocab::new(vec![(0, b"elsewhere".to_vec())]).with_special_tokens(1..80);
    Grammar::from_glrm(ONE).unwrap().bind_snapshot("T", TokenSet::from_ranges([5..=25])).unwrap().compile(&same_ids).unwrap();
}

#[test]
fn full_128k_and_nearly_full_vocabularies_compile_and_mask_compactly() {
    const N: u32 = 131_072;
    let vocab = Vocab::new(Vec::<(u32, Vec<u8>)>::new()).with_special_tokens(0..N);
    for excluded in [None, Some(65_535)] {
        let ranges = match excluded {
            None => vec![0..=N - 1],
            Some(id) => vec![0..=id - 1, id + 1..=N - 1],
        };
        let tokens = TokenSet::from_ranges(ranges.iter().cloned());
        assert_eq!(tokens.range_pairs().len(), if excluded.is_some() { 2 } else { 1 });
        for mode in MODES {
            let constraint = Grammar::from_glrm(ONE).unwrap().bind_snapshot("T", tokens.clone()).unwrap()
                .compile_with_build_options(&vocab, mode.build_options())
                .unwrap_or_else(|e| panic!("{mode:?}, excluded={excluded:?}: {e}"));
            let mask = constraint.start().mask();
            assert_eq!(mask.iter().map(|w| w.count_ones()).sum::<u32>(), N - u32::from(excluded.is_some()));
            for id in [0, 31, 32, 65_534, 65_535, 65_536, N - 1] {
                assert_eq!(allowed(&mask, id), excluded != Some(id));
                if excluded != Some(id) {
                    let mut state = constraint.start();
                    assert!(state.commit_token(id));
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
        let constraint = Grammar::from_glrm(source).unwrap()
            .bind_snapshot("A", TokenSet::from_ranges([8..=12])).unwrap()
            .bind_snapshot("B", TokenSet::from_ranges([10..=14])).unwrap()
            .compile_with_build_options(&vocab, mode.build_options()).unwrap();
        assert_mask(&constraint, &[], 20, |id| id == 0);
        assert_mask(&constraint, &[0], 20, |id| (8..=14).contains(&id));
        for exact in 8..=14 {
            assert_mask(&constraint, &[0, exact], 20,
                |id| id == 1 || id == 7 || (exact >= 10 && id == 2));
            let mut state = constraint.start();
            assert!(state.commit_token(0));
            let mut independent = state.clone();
            let before = independent.mask();
            assert!(state.commit_token(exact));
            assert_eq!(independent.mask(), before);
            // Byte aliases outside the set are not accepted as exact IDs.
            assert!(!independent.commit_token(6));
            assert!(state.commit_token(1));
            assert_eq!(state.is_accepting(), exact <= 12);
            if exact >= 10 {
                assert!(state.commit_token(1));
                assert!(state.is_accepting());
            }
        }
    }
}
#[test]
fn ranges_preserve_forcing_and_reserved_end_token_policy() {
    let vocab = Vocab::new(Vec::new()).with_special_tokens(0..16);
    for mode in MODES {
        let singleton = Grammar::from_glrm(ONE).unwrap()
            .bind_snapshot("T", TokenSet::from_ranges([5..=5])).unwrap()
            .compile_with_build_options(&vocab, mode.build_options()).unwrap();
        assert_eq!(singleton.start().forced_canonical_reference(), vec![5]);
        let repeated = Grammar::from_glrm("glrm 1; start start; extern token T; nt start = T T;").unwrap()
            .bind_snapshot("T", TokenSet::from_ranges([0..=15])).unwrap()
            .compile_with_build_options(&vocab, mode.build_options().end_tokens([8])).unwrap();
        assert_mask(&repeated, &[], 16, |id| id != 8);
        assert_mask(&repeated, &[7], 16, |id| id != 8);
        assert_mask(&repeated, &[7, 9], 16, |id| id == 8);
        let mut early = repeated.start();
        // An end token before a complete match is a failed commit.
        assert!(!early.commit_token(8));
        assert!(!early.is_accepting());
        assert!(early.mask().iter().all(|&word| word == 0));
        let mut state = repeated.start();
        for id in [7, 9, 8] { assert!(state.commit_token(id)); }
        assert!(state.is_accepting());
        assert!(state.mask().iter().all(|&word| word == 0));
    }
}

#[test]
fn neutral_scalar_module_formats_are_not_misread_as_range_artifacts() {
    let vocab = Vocab::new(Vec::new()).with_special_tokens(0..128);
    let module = Grammar::from_glrm(ONE).unwrap().prepare(&vocab).unwrap()
        .bind_snapshot("T", TokenSet::from_ranges([0..=127])).unwrap();
    let bytes = module.save();
    for version in [b"GLRMOD03", b"GLRMOD04", b"GLRMOD05", b"GLRMOD06"] {
        let mut old = bytes.clone();
        old[..8].copy_from_slice(version);
        assert!(PreparedGrammar::load(&old, &vocab).is_err());
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
            let grammar = Grammar::from_glrm(source).unwrap()
                .bind_snapshot("A", TokenSet::from_ranges([a])).unwrap()
                .bind_snapshot("B", TokenSet::from_ranges([b])).unwrap();
            let options = mode.build_options().end_tokens([15]);
            let constraint = grammar.compile_with_build_options(&vocab, options).unwrap();
            for constraint in [constraint.clone(), Constraint::load(constraint.save(), &vocab).unwrap()] {
                assert_mask(&constraint, &[0, 0], 16, |id| id == 15);
                let mut state = constraint.start();
                assert!(state.commit_token(0));
                assert!(state.commit_token(0));
                assert!(allowed(&state.mask(), 15), "{mode:?}");
                assert!(state.commit_token(15));
                assert!(state.mask().iter().all(|word| *word == 0), "{mode:?}");
                assert!(!state.commit_token(0), "{mode:?}");
            }
        }
    }
}

#[test]
fn full_empty_spelled_ranges_stay_exact_across_grammar_positions() {
    let n = 32_768u32;
    let vocab = Vocab::new((0..n).map(|id| (id, Vec::new())).collect());
    for mode in MODES {
        let grammar = Grammar::from_glrm("glrm 1; start start; extern token A; extern token B; nt start = A B;").unwrap()
            .bind_snapshot("A", TokenSet::from_ranges([0..=n - 1])).unwrap()
            .bind_snapshot("B", TokenSet::from_ranges([0..=n / 2 - 1, n / 2 + 1..=n - 1])).unwrap();
        let original = grammar.compile_with_build_options(&vocab,
            mode.build_options().end_tokens([n])).unwrap();
        for constraint in [original.clone(), Constraint::load(original.save(), &vocab).unwrap()] {
            assert_mask(&constraint, &[], n + 1, |id| id < n);
            assert_mask(&constraint, &[0], n + 1, |id| id < n && id != n / 2);
            assert_mask(&constraint, &[0, n - 1], n + 1, |id| id == n);
        }
    }
}
