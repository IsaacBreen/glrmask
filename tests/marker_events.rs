//! Public marker integration gates. The engine reports observations, not captures.
use glrmask::{BuildOptions, Constraint, Grammar, MaskStrategy, PreparedGrammar, Vocab};

fn modes() -> [MaskStrategy; 4] {
    [MaskStrategy::TokenTrie, MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa, MaskStrategy::Auto]
}
fn options(mode: MaskStrategy) -> BuildOptions { BuildOptions::default().optimization(mode) }
fn vocabulary() -> Vocab {
    Vocab::new(vec![(0,b"a".to_vec()), (1,b"b".to_vec()), (2,b"ab".to_vec()),
        (3,b"x".to_vec()), (4,b"y".to_vec()), (5,b"xab y".to_vec()),
        (6,b"xaby".to_vec()), (7,b"pxabyq".to_vec()), (8,b"p".to_vec()),
        (9,b"q".to_vec()), (10,Vec::new()), (11,b"<event>".to_vec())])
}
fn annotated() -> Grammar {
    Grammar::from_glrm(r#"glrm 1; start root; nt root = @mark(10) "a" @mark(20) "b" @mark(30);"#)
}
fn equivalent_replay(a: &Constraint, b: &Constraint, tokens: &[u32]) {
    let compare_events = a.marker_ids() == b.marker_ids();
    let mut a = a.start(); let mut b = b.start();
    assert_eq!(a.mask(), b.mask());
    if compare_events { assert_eq!(a.marker_events(), b.marker_events()); }
    for &token in tokens {
        assert_eq!(a.commit_token(token).is_ok(), b.commit_token(token).is_ok());
        assert_eq!(a.mask(), b.mask());
        assert_eq!(a.is_accepting(), b.is_accepting());
        assert_eq!(a.is_rejected(), b.is_rejected());
        if compare_events { assert_eq!(a.marker_events(), b.marker_events()); }
    }
}

#[test]
fn marker_inventory_roundtrips_every_public_mode_and_vocabulary_form() {
    let vocab = vocabulary();
    for mode in modes() {
        let fresh = annotated().compile_with_build_options(&vocab, options(mode)).unwrap();
        assert_eq!(fresh.marker_ids().len(), 3, "{mode:?}");
        assert_eq!(fresh.marker_ids(), &[10, 20, 30]);
        assert_eq!(batch(&fresh.start()), vec![(10, 0)]);
        let bytes = fresh.save();
        assert!(bytes.starts_with(b"GLRMARK1"));
        let owned = Constraint::load(bytes.clone()).unwrap();
        let borrowed = Constraint::load(bytes.as_slice()).unwrap();
        let bound = Constraint::load_with_vocab(bytes.clone(), &vocab).unwrap();
        let external = fresh.save_without_vocab().unwrap();
        assert!(Constraint::load(external.clone()).is_err());
        let external = Constraint::load_with_vocab(external, &vocab).unwrap();
        for loaded in [&owned, &borrowed, &bound, &external] {
            assert_eq!(fresh.marker_ids(), loaded.marker_ids(), "{mode:?}");
            for input in [&[0,1][..], &[2][..]] { equivalent_replay(&fresh, loaded, input); }
            let twice = Constraint::load(loaded.save()).unwrap();
            assert_eq!(fresh.marker_ids(), twice.marker_ids());
        }
        let plain = Grammar::from_ebnf(r#"start ::= "a" "b""#).compile_with_build_options(&vocab, options(mode)).unwrap();
        assert!(plain.marker_ids().is_empty());
        assert!(!plain.save().starts_with(b"GLRMARK1"));
        equivalent_replay(&plain, &fresh, &[0,1]);
        equivalent_replay(&plain, &fresh, &[2]);
    }
}

#[test]
fn prepared_and_nested_compiled_children_keep_typed_marker_inventory() {
    let vocab = vocabulary();
    let host = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = @mark(40) "x" child "y" @mark(50);"#)
        .prepare(&vocab).unwrap();
    let host = PreparedGrammar::load(host.save()).unwrap();
    for child_mode in modes() {
        let child = annotated().compile_with_build_options(&vocab, options(child_mode)).unwrap();
        let child = Constraint::load(child.save()).unwrap();
        let mut bound = host.clone();
        bound.bind("child", &child).unwrap();
        let bound = PreparedGrammar::load(bound.save()).unwrap();
        for mode in [MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa, MaskStrategy::Auto] {
            let linked = bound.link_with(options(mode)).unwrap();
            assert_eq!(linked.marker_ids().len(), 5, "child={child_mode:?} link={mode:?}");
            let loaded = Constraint::load(linked.save()).unwrap();
            let external = Constraint::load_with_vocab(linked.save_without_vocab().unwrap(), &vocab).unwrap();
            for value in [&linked,&loaded,&external] {
                assert_eq!(value.marker_ids(), linked.marker_ids());
                equivalent_replay(&linked,value,&[6]);
                let mut state=value.start();state.commit_token(6).unwrap();assert!(state.is_accepting());
                assert_eq!(batch(&state), vec![(10, 1), (20, 2), (30, 3), (50, 4)],
                    "child={child_mode:?} link={mode:?}");
            }
            let mut outer = Grammar::from_glrm(r#"glrm 1; start root; extern grammar child; nt root = @mark(60) "p" child "q" @mark(70);"#)
                .prepare(&vocab).unwrap();
            outer.bind("child", &loaded).unwrap();
            let outer=outer.link_with(options(mode)).unwrap();
            assert_eq!(outer.marker_ids().len(),7);
            let outer = Constraint::load(outer.save()).unwrap();
            assert_eq!(outer.marker_ids().len(),7);
            let mut state=outer.start();state.commit_token(7).unwrap();assert!(state.is_accepting());
            assert_eq!(batch(&state), vec![(40, 1), (10, 2), (20, 3), (30, 4), (50, 5), (70, 6)]);
        }
    }
}

#[test]
fn malformed_marker_envelopes_fail_without_panics() {
    let vocab=vocabulary();
    let bytes=annotated().compile_with_build_options(&vocab,options(MaskStrategy::TokenClassTrie)).unwrap().save();
    let mut malformed=Vec::new();
    for length in [0,1,7,8,12,19,20,27,28,bytes.len()-1] {malformed.push(bytes[..length].to_vec());}
    for count in [0,u32::MAX] {let mut v=bytes.clone();v[8..12].copy_from_slice(&count.to_le_bytes());malformed.push(v);}
    for length in [0,u64::MAX] {let mut v=bytes.clone();v[12..20].copy_from_slice(&length.to_le_bytes());malformed.push(v);}
    let mut duplicate=bytes.clone();let first=duplicate[20..24].to_vec();duplicate[28..32].copy_from_slice(&first);malformed.push(duplicate);
    let mut invalid=bytes.clone();invalid[20..24].copy_from_slice(&u32::MAX.to_le_bytes());malformed.push(invalid);
    // Terminal 1 is the ordinary "a" lexer terminal, not an observation probe.
    // It is in range and correctly ordered, but must still be rejected.
    let mut input_bound=bytes.clone();input_bound[20..24].copy_from_slice(&1u32.to_le_bytes());malformed.push(input_bound);
    let mut future=bytes.clone();future[7]=b'2';malformed.push(future);
    for (case,bytes) in malformed.into_iter().enumerate() {
        let result=std::panic::catch_unwind(|| Constraint::load(bytes));
        assert!(result.is_ok(),"malformed case {case} panicked");
        assert!(result.unwrap().is_err(),"malformed case {case} was accepted");
    }
}

fn batch(state: &glrmask::ConstraintState<'_>) -> Vec<(u32, u64)> {
    state.marker_events().iter().map(|event| (event.id, event.byte_offset)).collect()
}

#[test]
fn ignored_terminal_boundaries_remain_observable_inside_model_tokens() {
    let vocab = Vocab::new(vec![(0,b"a".to_vec()),(1,b" ".to_vec()),
        (2,b"b".to_vec()),(3,b"a  b".to_vec())]);
    let grammar = Grammar::from_glrm(r#"glrm 1; start root; ignore WS; t WS=" "; nt root="a" @mark(1) "b";"#);
    for mode in modes() {
        let constraint = grammar.compile_with_build_options(&vocab,options(mode)).unwrap();
        for tokens in [&[0,1,1,2][..], &[3][..]] {
            let mut state = constraint.start();
            let mut events = batch(&state);
            for &token in tokens { state.commit_token(token).unwrap(); events.extend(batch(&state)); }
            assert_eq!(events, vec![(1,1),(1,2),(1,3)], "{mode:?} {tokens:?}");
            assert!(state.is_accepting());
        }
    }
}

#[test]
fn nullable_marker_only_cycles_are_observations_not_runtime_transitions() {
    let vocab = Vocab::new(vec![(0,b"a".to_vec())]);
    for mode in modes() {
        let empty = Grammar::from_glrm("glrm 1; start root; nt root=@mark(7);")
            .compile_with_build_options(&vocab,options(mode)).unwrap();
        let state = empty.start();
        assert!(state.is_accepting());
        assert_eq!(batch(&state), vec![(7,0)]);
        let repeat = Grammar::from_glrm(r#"glrm 1; start root; nt root=@mark(7)* "a";"#)
            .compile_with_build_options(&vocab,options(mode)).unwrap();
        let repeat = Constraint::load(repeat.save()).unwrap();
        let mut state = repeat.start();
        assert_eq!(batch(&state), vec![(7,0)]);
        state.commit_token(0).unwrap();
        assert!(state.is_accepting());
        assert!(state.marker_events().is_empty());
    }
}

#[test]
fn recursive_and_repeated_components_keep_positions_without_capture_pairing() {
    let vocab = Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"aabb".to_vec())]);
    let grammar = Grammar::from_glrm(r#"glrm 1; start root; nt root="a" @mark(1) root? "b" @mark(2);"#);
    for mode in modes() {
        let constraint = grammar.compile_with_build_options(&vocab,options(mode)).unwrap();
        for tokens in [&[0,0,1,1][..], &[2][..]] {
            let mut state = constraint.start(); let mut events=batch(&state);
            for &token in tokens { state.commit_token(token).unwrap(); events.extend(batch(&state)); }
            assert_eq!(events, vec![(1,1),(1,2),(2,3),(2,4)], "{mode:?} {tokens:?}");
            assert!(state.is_accepting());
        }
    }
}
