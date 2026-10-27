//! Original-ID and byte-language forcing regressions. Kept separate from
//! backend-agreement tests: comparing two identical bugs is not an oracle.
use glrmask::test_modes::MaskMode;
use glrmask::{Constraint, Grammar, TokenSet, Vocab};
const MODES: [MaskMode; 6] = MaskMode::ALL;

fn allows(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|word| word & (1 << (id % 32)) != 0)
}
/// A rejected state cannot accept and has no allowed tokens. A live accepting
/// state legitimately has an empty mask when the grammar is complete.
fn assert_live(state: &glrmask::ConstraintState<'_>) {
    assert!(state.is_accepting() || !state.mask().iter().all(|word| *word == 0),
        "a successful commit left a rejected state");
}
fn replay(state: &glrmask::ConstraintState<'_>, ids: &[u32]) {
    let before=state.mask();let mut next=state.clone();
    for &id in ids {
        assert!(!next.is_accepting(), "forced beyond an accepting boundary");
        assert!(allows(&next.mask(),id), "forcing selected an unadmitted ID {id}");
        assert!(next.commit_token(id));assert_live(&next);
    }
    assert_eq!(state.mask(),before,"forcing/replay mutated source state");
}

fn checked_force(constraint: &Constraint) -> Vec<u32> {
    let state = constraint.start();
    let before = state.mask();
    let accepted = state.is_accepting();
    let ids = state.clone().commit_forced_tokens();
    assert_eq!(ids,state.forced_canonical_reference(),"public forcing diverged from canonical reference");
    assert_eq!(state.mask(), before, "forcing mutated the source mask");
    assert_eq!(state.is_accepting(), accepted);
    assert_live(&state);
    let mut replay = state.clone();
    for &id in &ids {
        assert!(!replay.is_accepting(), "forced past an accepting token boundary");
        assert!(allows(&replay.mask(), id), "forced ID {id} is excluded by the authoritative mask");
        assert!(replay.commit_token(id));
        assert_live(&replay);
    }
    ids
}

#[test]
fn forcing_stops_when_acceptance_competes_with_more_text() {
    let vocab = Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"c".to_vec())]);
    for mode in MODES {
        let c = Grammar::from_glrm("glrm 1; start root; nt root = \"ab\" \"c\"?;").unwrap()
            .compile_with_build_options(&vocab, mode.build_options()).unwrap();
        assert_eq!(checked_force(&c), [0,1], "{mode:?}");
        assert_eq!(checked_force(&Constraint::load(c.save(), &vocab).unwrap()), [0,1]);
    }
}

#[test]
fn exact_only_event_competes_with_text_even_without_spelling() {
    for empty_spelling in [false,true] {
        let mut entries=vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec())];
        if empty_spelling { entries.push((9,Vec::new())); }
        let vocab=Vocab::new(entries);
        for mode in MODES {
            let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = \"ab\" | MARK;").unwrap()
                .bind_snapshot("MARK",TokenSet::new([9])).unwrap()
                .compile_with_build_options(&vocab,mode.build_options()).unwrap();
            assert!(checked_force(&c).is_empty(), "{mode:?}, empty={empty_spelling}");
        }
    }
}

#[test]
fn same_spelling_exact_event_is_not_a_byte_prefix_alias() {
    let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(9,b"a".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = \"ab\" | MARK;").unwrap()
            .bind_snapshot("MARK",TokenSet::new([9])).unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        assert!(checked_force(&c).is_empty(), "{mode:?}");
    }
}

#[test]
fn singleton_mixed_id_keeps_ordinary_and_exact_successors() {
    let vocab=Vocab::new(vec![(10,b"a".to_vec()),(11,b"b".to_vec()),(12,b"c".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = \"ab\" | MARK \"c\";").unwrap()
            .bind_snapshot("MARK",TokenSet::new([10])).unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        assert_eq!(checked_force(&c), [10], "{mode:?}");
        let mut after=c.start();assert!(after.commit_token(10));
        assert!(allows(&after.mask(),11));assert!(allows(&after.mask(),12));
        assert!(after.forced_canonical_reference().is_empty());
    }
}

#[test]
fn canonical_alias_selection_excludes_reserved_root_end_ids() {
    let vocab=Vocab::new(vec![(0,b"x".to_vec()),(1,b"x".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"x\";").unwrap()
            .compile_with_build_options(&vocab,mode.build_options().end_tokens([0])).unwrap();
        assert_eq!(checked_force(&c),[1],"{mode:?}");
    }
}

#[test]
fn canonical_text_forcing_is_not_token_id_uniqueness() {
    let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(3,b"ab".to_vec()),(4,b"abc".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"ab\";").unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        // ID4 is not admitted. Its longer spelling must not block the stable ID2.
        assert_eq!(checked_force(&c),[2],"{mode:?}");
    }
}

#[test]
fn canonical_greedy_token_does_not_cross_an_intermediate_exact_event_boundary() {
    let vocab=Vocab::new(vec![
        (0,b"a".to_vec()),(1,b"ab".to_vec()),(2,b"b".to_vec()),
    ]);
    let ordinary=Grammar::from_glrm("glrm 1; start root; nt root = \"ab\";").unwrap();
    let guarded=Grammar::from_glrm(
        "glrm 1; start root; extern token MARK; nt root = \"ab\" | \"a\" MARK \"b\";",
    ).unwrap().bind_snapshot("MARK",TokenSet::new([9])).unwrap();
    for mode in MODES {
        let options=mode.build_options();
        let baseline=ordinary.compile_with_build_options(&vocab,options.clone()).unwrap();
        assert_eq!(checked_force(&baseline),[1],"ordinary baseline {mode:?}");

        let guarded=guarded.compile_with_build_options(&vocab,options).unwrap();
        for guarded in [guarded.clone(),Constraint::load(guarded.save(), &vocab).unwrap()] {
            // Both paths project to ordinary bytes `ab`, but greedy token 1
            // would cross the valid MARK opportunity after `a`. Returning token
            // 0 instead would cease to be the canonical greedy tokenization.
            assert!(checked_force(&guarded).is_empty(),"event boundary {mode:?}");
            let mask=guarded.start().mask();
            assert!(allows(&mask,0) && allows(&mask,1));
            let mut event_path=guarded.start();
            for id in [0,9,2] { assert!(allows(&event_path.mask(),id));assert!(event_path.commit_token(id)); }
            assert!(event_path.is_accepting());
        }
    }
}

#[test]
fn exact_events_before_and_after_text_keep_their_original_ids() {
    let vocab=Vocab::new(vec![
        (0,b"a".to_vec()),(1,b"ab".to_vec()),(2,b"b".to_vec()),(8,Vec::new()),
    ]);
    for (event,expected) in [
        (8,vec![8,1,8]),
        (9,vec![9,1,9]),
    ] {
        let grammar=Grammar::from_glrm(
            "glrm 1; start root; extern token BEFORE; extern token AFTER; nt root = BEFORE \"ab\" AFTER;",
        ).unwrap().bind_snapshot("BEFORE",TokenSet::new([event])).unwrap()
            .bind_snapshot("AFTER",TokenSet::new([event])).unwrap();
        for mode in MODES {
            let c=grammar.compile_with_build_options(&vocab,mode.build_options()).unwrap();
            assert_eq!(checked_force(&c),expected,"event={event}, {mode:?}");
        }
    }
    let ambiguous=Grammar::from_glrm("extern token MARK start root nt root=MARK \"ab\"").unwrap()
        .bind_snapshot("MARK",TokenSet::new([9,10])).unwrap();
    for mode in MODES {
        let c=ambiguous.compile_with_build_options(&vocab,mode.build_options()).unwrap();
        assert!(checked_force(&c).is_empty(),"multiple exact IDs {mode:?}");
    }
}

#[test]
fn unicode_forcing_does_not_need_individual_utf8_byte_tokens() {
    let vocab=Vocab::new(vec![(0,"\u{e9}".as_bytes().to_vec()),(1,"\u{e9}\u{e9}".as_bytes().to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"\u{e9}\u{e9}\";").unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        assert_eq!(checked_force(&c),[1],"{mode:?}");
    }
}

#[test]
fn longer_admitted_token_can_block_a_canonical_prefix_at_acceptance() {
    let vocab=Vocab::new(vec![(0,b"ab".to_vec()),(1,b"abc".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"ab\" \"c\"?;").unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        assert!(checked_force(&c).is_empty(),"{mode:?}");
    }
}

#[test]
fn exact_event_chain_retains_ids_and_never_canonicalizes_equal_bytes() {
    let vocab=Vocab::new(vec![(0,b"x".to_vec()),(1,b"x".to_vec()),(2,Vec::new())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token A; extern token B; extern token C; nt root = A B C;").unwrap()
            .bind_snapshot("A",TokenSet::new([1])).unwrap()
            .bind_snapshot("B",TokenSet::new([2])).unwrap()
            .bind_snapshot("C",TokenSet::new([3])).unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        assert_eq!(checked_force(&c),[1,2,3],"{mode:?}");
    }
}


/// Independent finite-language oracle; no tokenizer/parser implementation is
/// used to decide the expected forcing result.
fn finite_language_force(words: &[&[u8]], vocab: &[(u32,Vec<u8>)]) -> Vec<u32> {
    let mut suffixes=words.iter().map(|word| word.to_vec()).collect::<Vec<_>>();
    let mut result=Vec::new();
    loop {
        if suffixes.iter().any(Vec::is_empty) { return result; }
        let first=&suffixes[0];
        let width=(0..first.len()).take_while(|&i| suffixes.iter().all(|word| word.get(i)==first.get(i))).count();
        let prefix=&first[..width];
        if prefix.is_empty() { return result; }
        let admitted=vocab.iter().filter(|(_,bytes)| !bytes.is_empty() && suffixes.iter().any(|word| word.starts_with(bytes)));
        let mut best=None;
        for (id,bytes) in admitted {
            if bytes.len()>prefix.len() && bytes.starts_with(prefix) { return result; }
            if prefix.starts_with(bytes) && best.is_none_or(|(old_id,old_width)| bytes.len()>old_width || (bytes.len()==old_width && *id<old_id)) {
                best=Some((*id,bytes.len()));
            }
        }
        let Some((id,width))=best else { return result; };
        result.push(id);
        for word in &mut suffixes { word.drain(..width); }
    }
}

#[test]
fn finite_byte_languages_match_an_independent_forcing_oracle() {
    let entries=vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"aa".to_vec()),
        (3,b"ab".to_vec()),(4,b"ba".to_vec()),(5,b"bb".to_vec()),(6,b"aba".to_vec()),(7,b"ab".to_vec())];
    let vocabulary=Vocab::new(entries.clone());
    let words: [&[u8];5]=[b"a",b"ab",b"aba",b"abb",b"baa"];
    for subset in 1..(1<<words.len()) {
        let language=words.iter().enumerate().filter(|(i,_)| subset & (1<<i)!=0).map(|(_,word)|*word).collect::<Vec<_>>();
        let alternatives=language.iter().map(|word| format!("\"{}\"",std::str::from_utf8(word).unwrap())).collect::<Vec<_>>().join(" | ");
        let source=format!("glrm 1; start root; nt root = {alternatives};");
        let expected=finite_language_force(&language,&entries);
        for mode in MODES {
            let c=Grammar::from_glrm(&source).unwrap().compile_with_build_options(&vocabulary,mode.build_options()).unwrap();
            assert_eq!(checked_force(&c),expected,"{mode:?} {source}");
        }
    }
}


#[test]
fn strict_id_forcing_is_distinct_from_canonical_byte_fastforward() {
    let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(3,b"ab".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"ab\";").unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        let state=c.start();let before=state.mask();
        assert_eq!(state.forced_canonical_reference(),[2]);
        assert!(state.forced_token_ids_reference().is_empty());
        assert_eq!(state.mask(),before);
        let mut cursor=state.clone();assert!(cursor.commit_token(0));
        assert_eq!(cursor.forced_token_ids_reference(),[1]);
        let loaded=Constraint::load(c.save(), &vocab).unwrap();
        assert!(loaded.start().forced_token_ids_reference().is_empty());
    }
}

#[test]
fn strict_id_forcing_keeps_exact_id_events_and_stops_at_acceptance() {
    let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = MARK \"a\" \"b\"?;").unwrap()
            .bind_snapshot("MARK",TokenSet::new([9])).unwrap()
            .compile_with_build_options(&vocab,mode.build_options()).unwrap();
        let state=c.start();let mask=state.mask();
        assert_eq!(state.forced_token_ids_reference(),[9,0]);
        assert_eq!(state.mask(),mask);
        let mut replay=state.clone();
        for id in state.forced_token_ids_reference() {assert!(allows(&replay.mask(),id));assert!(replay.commit_token(id));}
        assert!(replay.is_accepting());assert!(replay.forced_token_ids_reference().is_empty());
    }
}

#[test]
fn utf8_partial_token_frontier_keeps_exact_event_alternative_distinct() {
    let v=Vocab::new(vec![(0,vec![0xC3]),(1,vec![0xA9]),(2,b"x".to_vec()),
        (3,b"y".to_vec()),(4,"éx".as_bytes().to_vec())]);
    let ordinary=Grammar::from_glrm("start root t TEXT=/éx/ nt root=TEXT").unwrap();
    let dual=Grammar::from_glrm("extern token MARK start root t TEXT=/éx/ nt root=TEXT | MARK \"y\"").unwrap()
        .bind_snapshot("MARK",TokenSet::new([0])).unwrap();
    for mode in MODES {
        let c=ordinary.compile_with_build_options(&v,mode.build_options()).unwrap();
        let mut s=c.start();assert!(s.commit_token(0));
        assert_eq!(s.forced_canonical_reference(),[1,2]);assert_eq!(s.forced_token_ids_reference(),[1,2]);
        replay(&s,&s.forced_canonical_reference());
        let c=dual.compile_with_build_options(&v,mode.build_options()).unwrap();
        let mut s=c.start();assert!(s.forced_canonical_reference().is_empty());assert!(s.commit_token(0));
        assert!(allows(&s.mask(),1));assert!(allows(&s.mask(),3));
        assert!(s.forced_canonical_reference().is_empty());assert!(s.forced_token_ids_reference().is_empty());
        for path in [&[1,2][..],&[3][..]] {
            let mut t=s.clone();for id in path {assert!(t.commit_token(*id));}assert!(t.is_accepting());
        }
    }
}

#[test]
fn child_trailing_ignore_and_parent_suffix_do_not_force_past_exact_event() {
    let v=Vocab::new(vec![(0,b"P".to_vec()),(1,b"a".to_vec()),
        (2,b" ".to_vec()),(3,b" X".to_vec()),(4,b"X".to_vec()),(5,b"Pa X".to_vec()),
        (6,b"\t".to_vec())]);
    let child=Grammar::from_glrm("extern token MARK start item ignore SPACE t SPACE=/ +/ nt item=\"a\" MARK?").unwrap();
    let parent=Grammar::from_glrm("extern grammar CHILD start root nt root=\"P\" CHILD \" X\"").unwrap();
    let p=parent.prepare(&v).unwrap().bind_snapshot("CHILD",child.prepare(&v).unwrap()).unwrap()
        .bind_snapshot("CHILD::MARK",TokenSet::new([9])).unwrap();
    for mode in MODES {
        let c=p.link_with(mode.build_options()).unwrap();
        for path in [&[5][..],&[0,1,3][..],&[0,2,1,2,3][..],&[0,1,9,3][..]] {
            let mut s=c.start();
            for id in path {
                replay(&s,&s.forced_canonical_reference());replay(&s,&s.forced_token_ids_reference());
                assert!(allows(&s.mask(),*id),"{mode:?}, path={path:?}, id={id}");assert!(s.commit_token(*id));
            }
            assert!(s.is_accepting(),"{mode:?}, path={path:?}");assert!(s.forced_canonical_reference().is_empty());
        }
        let mut s=c.start();assert!(s.commit_token(0));assert!(s.commit_token(1));
        assert!(allows(&s.mask(),9));assert!(allows(&s.mask(),3));
        assert!(s.forced_canonical_reference().is_empty(),"byte proof must stop before optional exact event");
        assert!(!allows(&s.mask(),6),"child's space ignore must not allow arbitrary whitespace");
    }
}

fn long_forced_fixture() -> (Constraint, Vocab) {
    let vocab = Vocab::new(vec![
        (0, b"first ".to_vec()),
        (1, b"second ".to_vec()),
        (2, b"third ".to_vec()),
        (3, b"fourth ".to_vec()),
        (4, b"end".to_vec()),
    ]);
    let grammar = Grammar::from_glrm("glrm 1; start root; nt root = \"first second third fourth end\";").unwrap();
    let constraint = grammar.compile(&vocab).unwrap();
    (constraint, vocab)
}

#[test]
fn commit_forced_tokens_semantics_match_commit_tokens_replay() {
    let (constraint, _) = long_forced_fixture();
    let mut state = constraint.start();
    let mut replay = state.clone();
    let returned = state.commit_forced_tokens();
    assert!(!returned.is_empty());
    assert!(replay.commit_tokens(&returned));
    assert_eq!(state.mask(), replay.mask());
    assert_eq!(state.is_accepting(), replay.is_accepting());
    assert_eq!(state.is_terminated(), replay.is_terminated());
}

#[test]
fn commit_forced_tokens_stops_after_1000_tokens_and_continues() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
    let parts = (0..1005).map(|_| glrmask::Expr::literal("a"));
    let grammar = Grammar::from_expr(glrmask::Expr::sequence(parts)).unwrap();
    let constraint = grammar.compile(&vocab).unwrap();
    let mut state = constraint.start();
    let first = state.commit_forced_tokens();
    assert_eq!(first.len(), 1000);
    assert!(!state.is_accepting());
    let second = state.commit_forced_tokens();
    assert_eq!(second.len(), 5);
    assert!(state.is_accepting());
}

#[test]
fn code_bound_token_set_with_unregistered_id_is_allowed_after_literal() {
    let vocab = Vocab::new(vec![(0, b"hello ".to_vec())]);
    assert!(!vocab.contains_token_id(99));
    let mut module = glrmask::Module::new(&Grammar::from_glrm("glrm 1; start root; extern token SPECIAL; nt root = \"hello \" SPECIAL;").unwrap());
    module.bind("SPECIAL", TokenSet::new([99])).unwrap();
    let constraint = module.compile(&vocab).unwrap();
    assert!(constraint.mask_len() * 32 > 99);
    let mut state = constraint.start();
    assert!(!allows(&state.mask(), 99));
    assert!(state.commit_token(0));
    assert!(allows(&state.mask(), 99));
    assert!(state.commit_token(99));
    assert!(state.is_accepting());
}
