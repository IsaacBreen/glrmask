//! Original-ID and byte-language forcing regressions. Kept separate from
//! backend-agreement tests: comparing two identical bugs is not an oracle.
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, Vocab};
const MODES: [Optimization; 4] = [Optimization::Auto, Optimization::FastBuild, Optimization::Balanced, Optimization::FastRuntime];

fn allows(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|word| word & (1 << (id % 32)) != 0)
}
fn replay(state: &glrmask::ConstraintState<'_>, ids: &[u32]) {
    let before=state.mask();let mut next=state.clone();
    for &id in ids {
        assert!(!next.is_accepting(), "forced beyond an accepting boundary");
        assert!(allows(&next.mask(),id), "forcing selected an unadmitted ID {id}");
        next.commit_token(id).unwrap();assert!(!next.is_rejected());
    }
    assert_eq!(state.mask(),before,"forcing/replay mutated source state");
}

fn checked_force(constraint: &Constraint) -> Vec<u32> {
    let state = constraint.start();
    let before = state.mask();
    let accepted = state.is_accepting();
    let ids = state.forced();
    assert_eq!(state.mask(), before, "forcing mutated the source mask");
    assert_eq!(state.is_accepting(), accepted);
    assert!(!state.is_rejected());
    let mut replay = state.clone();
    for &id in &ids {
        assert!(!replay.is_accepting(), "forced past an accepting token boundary");
        assert!(allows(&replay.mask(), id), "forced ID {id} is excluded by the authoritative mask");
        replay.commit_token(id).unwrap();
        assert!(!replay.is_rejected());
    }
    ids
}

#[test]
fn forcing_stops_when_acceptance_competes_with_more_text() {
    let vocab = Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"c".to_vec())]);
    for mode in MODES {
        let c = Grammar::from_glrm("glrm 1; start root; nt root = \"ab\" \"c\"?;")
            .compile_with(&vocab, BuildOptions::default().optimization(mode)).unwrap();
        assert_eq!(checked_force(&c), [0,1], "{mode:?}");
        assert_eq!(checked_force(&Constraint::load(&c.save()).unwrap()), [0,1]);
    }
}

#[test]
fn exact_only_event_competes_with_text_even_without_spelling() {
    for empty_spelling in [false,true] {
        let mut entries=vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec())];
        if empty_spelling { entries.push((9,Vec::new())); }
        let vocab=Vocab::new_with_exact_token_ids(entries,[9]);
        for mode in MODES {
            let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = \"ab\" | MARK;")
                .bind("MARK",vocab.tokens([9]).unwrap()).unwrap()
                .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
            assert!(checked_force(&c).is_empty(), "{mode:?}, empty={empty_spelling}");
        }
    }
}

#[test]
fn same_spelling_exact_event_is_not_a_byte_prefix_alias() {
    let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(9,b"a".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = \"ab\" | MARK;")
            .bind("MARK",vocab.tokens([9]).unwrap()).unwrap()
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        assert!(checked_force(&c).is_empty(), "{mode:?}");
    }
}

#[test]
fn singleton_mixed_id_keeps_ordinary_and_exact_successors() {
    let vocab=Vocab::new(vec![(10,b"a".to_vec()),(11,b"b".to_vec()),(12,b"c".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = \"ab\" | MARK \"c\";")
            .bind("MARK",vocab.tokens([10]).unwrap()).unwrap()
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        assert_eq!(checked_force(&c), [10], "{mode:?}");
        let mut after=c.start();after.commit_token(10).unwrap();
        assert!(allows(&after.mask(),11));assert!(allows(&after.mask(),12));
        assert!(after.forced().is_empty());
    }
}

#[test]
fn canonical_alias_selection_excludes_reserved_root_end_ids() {
    let vocab=Vocab::new(vec![(0,b"x".to_vec()),(1,b"x".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"x\";")
            .compile_with(&vocab,BuildOptions::default().optimization(mode).end_tokens([0])).unwrap();
        assert_eq!(checked_force(&c),[1],"{mode:?}");
    }
}

#[test]
fn canonical_text_forcing_is_not_token_id_uniqueness() {
    let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(3,b"ab".to_vec()),(4,b"abc".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"ab\";")
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        // ID4 is not admitted. Its longer spelling must not block the stable ID2.
        assert_eq!(checked_force(&c),[2],"{mode:?}");
    }
}

#[test]
fn unicode_forcing_does_not_need_individual_utf8_byte_tokens() {
    let vocab=Vocab::new(vec![(0,"\u{e9}".as_bytes().to_vec()),(1,"\u{e9}\u{e9}".as_bytes().to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"\u{e9}\u{e9}\";")
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        assert_eq!(checked_force(&c),[1],"{mode:?}");
    }
}

#[test]
fn longer_admitted_token_can_block_a_canonical_prefix_at_acceptance() {
    let vocab=Vocab::new(vec![(0,b"ab".to_vec()),(1,b"abc".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"ab\" \"c\"?;")
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        assert!(checked_force(&c).is_empty(),"{mode:?}");
    }
}

#[test]
fn exact_event_chain_retains_ids_and_never_canonicalizes_equal_bytes() {
    let vocab=Vocab::new_with_exact_token_ids(vec![(0,b"x".to_vec()),(1,b"x".to_vec()),(2,Vec::new())],[3]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token A; extern token B; extern token C; nt root = A B C;")
            .bind("A",vocab.tokens([1]).unwrap()).unwrap()
            .bind("B",vocab.tokens([2]).unwrap()).unwrap()
            .bind("C",vocab.tokens([3]).unwrap()).unwrap()
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
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
            let c=Grammar::from_glrm(&source).compile_with(&vocabulary,BuildOptions::default().optimization(mode)).unwrap();
            assert_eq!(checked_force(&c),expected,"{mode:?} {source}");
        }
    }
}


#[test]
fn strict_id_forcing_is_distinct_from_canonical_byte_fastforward() {
    let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(3,b"ab".to_vec())]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; nt root = \"ab\";")
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        let state=c.start();let before=state.mask();
        assert_eq!(state.forced(),[2]);
        assert!(state.forced_token_ids().is_empty());
        assert_eq!(state.mask(),before);
        let mut cursor=state.clone();cursor.commit_token(0).unwrap();
        assert_eq!(cursor.forced_token_ids(),[1]);
        let loaded=Constraint::load(&c.save()).unwrap();
        assert!(loaded.start().forced_token_ids().is_empty());
    }
}

#[test]
fn strict_id_forcing_keeps_exact_id_events_and_stops_at_acceptance() {
    let vocab=Vocab::new_with_exact_token_ids(vec![(0,b"a".to_vec()),(1,b"b".to_vec())],[9]);
    for mode in MODES {
        let c=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = MARK \"a\" \"b\"?;")
            .bind("MARK",vocab.tokens([9]).unwrap()).unwrap()
            .compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        let state=c.start();let mask=state.mask();
        assert_eq!(state.forced_token_ids(),[9,0]);
        assert_eq!(state.mask(),mask);
        let mut replay=state.clone();
        for id in state.forced_token_ids() {assert!(allows(&replay.mask(),id));replay.commit_token(id).unwrap();}
        assert!(replay.is_accepting());assert!(replay.forced_token_ids().is_empty());
    }
}

#[test]
fn utf8_partial_token_frontier_keeps_exact_event_alternative_distinct() {
    let v=Vocab::new(vec![(0,vec![0xC3]),(1,vec![0xA9]),(2,b"x".to_vec()),
        (3,b"y".to_vec()),(4,"éx".as_bytes().to_vec())]);
    let ordinary=Grammar::from_glrm("start root t TEXT=/éx/ nt root=TEXT");
    let dual=Grammar::from_glrm("extern token MARK start root t TEXT=/éx/ nt root=TEXT | MARK \"y\"")
        .bind("MARK",v.tokens([0]).unwrap()).unwrap();
    for mode in MODES {
        let c=ordinary.compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();
        let mut s=c.start();s.commit_token(0).unwrap();
        assert_eq!(s.forced(),[1,2]);assert_eq!(s.forced_token_ids(),[1,2]);
        replay(&s,&s.forced());
        let c=dual.compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();
        let mut s=c.start();assert!(s.forced().is_empty());s.commit_token(0).unwrap();
        assert!(allows(&s.mask(),1));assert!(allows(&s.mask(),3));
        assert!(s.forced().is_empty());assert!(s.forced_token_ids().is_empty());
        for path in [&[1,2][..],&[3][..]] {
            let mut t=s.clone();for id in path {t.commit_token(*id).unwrap();}assert!(t.is_accepting());
        }
    }
}

#[test]
fn child_trailing_ignore_and_parent_suffix_do_not_force_past_exact_event() {
    let v=Vocab::new_with_exact_token_ids(vec![(0,b"P".to_vec()),(1,b"a".to_vec()),
        (2,b" ".to_vec()),(3,b" X".to_vec()),(4,b"X".to_vec()),(5,b"Pa X".to_vec()),
        (6,b"\t".to_vec())],[9]);
    let child=Grammar::from_glrm("extern token MARK start item ignore SPACE t SPACE=/ +/ nt item=\"a\" MARK?");
    let parent=Grammar::from_glrm("extern grammar CHILD start root nt root=\"P\" CHILD \" X\"");
    let p=parent.prepare(&v).unwrap().bind("CHILD",child.prepare(&v).unwrap()).unwrap()
        .bind("CHILD::MARK",v.tokens([9]).unwrap()).unwrap();
    for mode in MODES {
        let c=p.link_with(BuildOptions::default().optimization(mode)).unwrap();
        for path in [&[5][..],&[0,1,3][..],&[0,2,1,2,3][..],&[0,1,9,3][..]] {
            let mut s=c.start();
            for id in path {
                replay(&s,&s.forced());replay(&s,&s.forced_token_ids());
                assert!(allows(&s.mask(),*id),"{mode:?}, path={path:?}, id={id}");s.commit_token(*id).unwrap();
            }
            assert!(s.is_accepting(),"{mode:?}, path={path:?}");assert!(s.forced().is_empty());
        }
        let mut s=c.start();s.commit_token(0).unwrap();s.commit_token(1).unwrap();
        assert!(allows(&s.mask(),9));assert!(allows(&s.mask(),3));
        assert!(s.forced().is_empty(),"byte proof must stop before optional exact event");
        assert!(!allows(&s.mask(),6),"child's space ignore must not allow arbitrary whitespace");
    }
}
