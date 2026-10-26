use glrmask::test_modes::MaskMode;
use glrmask::{Constraint, Grammar, MarkerEvent, TokenSet, Vocab};

const MODES: [MaskMode; 6] = MaskMode::ALL;
fn vocab(parts: &[&[u8]]) -> Vocab {
    Vocab::new(parts.iter().enumerate().map(|(id, bytes)| (id as u32, bytes.to_vec())).collect())
}
fn events(state: &glrmask::ConstraintState) -> Vec<(u32,u64)> {
    state.marker_events().iter().map(|e| (e.id,e.byte_offset)).collect()
}
fn compile(source: &str, vocab: &Vocab, mode: MaskMode) -> Constraint {
    Grammar::from_glrm(source).unwrap().compile_with_build_options(vocab, mode.build_options()).unwrap()
}

#[test]
fn initial_split_and_inside_fused_token_events_all_modes() {
    let v = vocab(&[b"a",b"b",b"ab"]);
    for mode in MODES {
        let c = compile("start root\nnt root = @mark(0) \"a\" @mark(7) \"b\" @mark(42)\n", &v, mode);
        for c in [&c, &Constraint::load(c.save(), &v).unwrap()] {
            let mut s = c.start();
            assert_eq!(events(&s), [(0,0)], "start {mode:?}");
            assert!(s.commit_token(0));
            assert_eq!(events(&s), [(7,1)], "a {mode:?}");
            assert!(s.commit_token(1));
            assert_eq!(events(&s), [(42,2)], "b {mode:?}");
            assert!(s.is_accepting());
            let mut fused = c.start();
            assert!(fused.commit_token(2));
            assert_eq!(events(&fused), [(7,1),(42,2)], "ab {mode:?}");
            let mut raw = c.start(); assert!(raw.commit_bytes(b"ab"));
            assert_eq!(events(&raw), events(&fused));
        }
    }
}

#[test]
fn partial_lexemes_do_not_observe_stale_parser_boundary() {
    let v = vocab(&[b"he",b"llo:",b"hello:"]);
    for mode in MODES {
        let c = compile("start root\nt WORD = \"hello\"\nnt root = @mark(0) WORD @mark(1) \":\" @mark(2)\n", &v, mode);
        let mut s=c.start(); assert_eq!(events(&s),[(0,0)]);
        assert!(s.commit_token(0)); assert!(events(&s).is_empty(), "partial {mode:?}");
        assert!(s.commit_token(1)); assert_eq!(events(&s),[(1,5),(2,6)], "suffix {mode:?}");
        let mut fused=c.start(); assert!(fused.commit_token(2));
        assert_eq!(events(&fused),events(&s));
    }
}

#[test]
fn repeated_same_ids_are_set_observations_not_a_parse_sequence() {
    let v = vocab(&[b"a",b"aa",b"aaaa"]);
    for mode in MODES {
        let c=compile("start root\nnt root = (@mark(10) \"a\" @mark(20))*\n",&v,mode);
        let mut s=c.start();assert_eq!(events(&s),[(10,0)]);
        assert!(s.commit_token(2));
        let expected=(1..=4).flat_map(|p|[(10,p),(20,p)]).collect::<Vec<_>>();
        assert_eq!(events(&s),expected,"four letters {mode:?}");
        assert!(s.commit_token(0));assert_eq!(events(&s),[(10,5),(20,5)]);
    }
}

#[test]
fn ambiguity_unions_observations_and_does_not_retract_dead_paths() {
    let v=vocab(&[b"a",b"b",b"c",b"ac"]);
    for mode in MODES {
        let c=compile("start root\nnt root = \"a\" @mark(7) \"b\" | \"a\" @mark(8) \"c\"\n",&v,mode);
        let mut s=c.start();assert!(s.commit_token(0));
        let retained=events(&s); assert_eq!(retained,[(7,1),(8,1)],"{mode:?}");
        assert!(s.commit_token(2));assert!(s.is_accepting());assert!(events(&s).is_empty());
        assert_eq!(retained,[(7,1),(8,1)]);
    }
}

#[test]
fn markers_at_same_offset_deduplicate_and_sparse_ids_are_not_array_sizes() {
    let v=vocab(&[b"x"]);
    for mode in MODES {
        let c=compile("start root\nnt root = @mark(4294967295) @mark(2) @mark(2) \"x\" @mark(2)\n",&v,mode);
        assert_eq!(c.marker_ids(),[2,u32::MAX]);
        let mut s=c.start();assert_eq!(events(&s),[(2,0),(u32::MAX,0)]);
        assert!(s.commit_token(0));assert_eq!(events(&s),[(2,1)]);
    }
}

#[test]
fn offsets_are_bytes_including_partial_utf8_tokens() {
    let v=vocab(&[b"\xc3",b"\xa9!",b"\xc3\xa9!"]);
    for mode in MODES {
        let c=compile("start root\nnt root = \"é\" @mark(1) \"!\" @mark(2)\n",&v,mode);
        let mut s=c.start();assert!(s.commit_token(0));assert!(events(&s).is_empty());
        assert!(s.commit_token(1));assert_eq!(events(&s),[(1,2),(2,3)],"{mode:?}");
        let mut fused=c.start();assert!(fused.commit_token(2));assert_eq!(events(&s),events(&fused));
    }
}

#[test]
fn clone_masks_forcing_errors_and_eos_obey_batch_lifecycle() {
    let v=vocab(&[b"a",b"b",b"ab",b"BAD",b"<eos>"]);
    for mode in MODES {
        let c=Grammar::from_glrm("start root\nnt root = @mark(0) \"a\" @mark(1) \"b\" @mark(2)\n").unwrap()
            .compile_with_build_options(&v,mode.build_options().end_tokens([4])).unwrap();
        let mut s=c.start();let initial=s.marker_events().to_vec();
        let _=s.mask();let _=s.forced_tokens();assert_eq!(s.marker_events(),initial);
        let mut fork=s.clone();assert!(fork.commit_token(2));
        assert_eq!(events(&fork),[(1,1),(2,2)]);assert_eq!(s.marker_events(),initial);
        assert!(s.commit_token(0));let observed=s.marker_events().to_vec();
        let _=s.mask();let _=s.forced_tokens();assert_eq!(s.marker_events(),observed);
        // A failed commit kills the state, so continue from a clone taken first.
        let mut rejected_batch=s.clone();
        assert!(!rejected_batch.commit_token(99999));
        assert!(events(&rejected_batch).is_empty());
        assert!(rejected_batch.mask().iter().all(|word| *word == 0));
        assert!(s.commit_token(1));assert_eq!(events(&s),[(2,2)]);
        assert!(s.commit_token(4));assert!(s.is_accepting());assert!(events(&s).is_empty());
        assert!(!s.commit_token(0));assert!(events(&s).is_empty());
        let mut rejected=c.start();let _=rejected.commit_token(3);
        assert!((rejected.mask().iter().all(|word| *word == 0)));assert!(events(&rejected).is_empty());
    }
}

#[test]
fn exact_model_events_and_marker_ids_are_distinct_namespaces() {
    for spelling in [b"<s>".as_slice(),b""] {
        let v=vocab(&[b"a",b"b",spelling]);
        for mode in MODES {
            let mut grammar=Grammar::from_glrm("start root\nextern token SEP\nnt root = \"a\" @mark(1) SEP @mark(2) \"b\" @mark(3)\n").unwrap()
                ;
            grammar.bind("SEP", TokenSet::new([2])).unwrap();
            let c=grammar.compile_with_build_options(&v,mode.build_options()).unwrap();
            let mut s=c.start();assert!(s.commit_token(0));assert_eq!(events(&s),[(1,1)]);
            assert!(s.commit_token(2));assert_eq!(events(&s),[(2,1+spelling.len() as u64)],"special {mode:?}");
            assert!(s.commit_token(1));assert_eq!(events(&s),[(3,2+spelling.len() as u64)]);
        }
    }
}

#[test]
fn public_event_is_copy_and_contains_no_per_event_string() {
    let e=MarkerEvent{id:7,byte_offset:12};let copied=e;assert_eq!(e,copied);
    assert!(std::mem::size_of::<MarkerEvent>()<=16);
}
