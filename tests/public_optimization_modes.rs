//! Public root policy and state operations across actual Dynamic, O2, and Static.
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, Vocab};

const MODES: [Optimization; 4] = [Optimization::Auto, Optimization::FastBuild,
    Optimization::Balanced, Optimization::FastRuntime];
fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|w| w & (1 << (id % 32)) != 0)
}
fn vocab() -> Vocab {
    Vocab::new(vec![(0,b"(".to_vec()),(2,b")".to_vec()),(7,b"()".to_vec()),
        (15,b"(".to_vec()),(20,b"x".to_vec()),(25,b"".to_vec())])
}

#[test]
fn recursive_masks_root_eos_force_clone_and_reload_agree() {
    let v = vocab();
    let g = Grammar::from_ebnf(r#"start ::= "(" start ")" start | """#);
    let reference = g.compile_with(&v, BuildOptions::default().end_tokens([100])).unwrap();
    for mode in MODES {
        let compiled = g.compile_with(&v, BuildOptions::default().optimization(mode).end_tokens([100])).unwrap();
        for c in [compiled.clone(), Constraint::load(compiled.save()).unwrap(),
            Constraint::load_with_vocab(compiled.save_without_vocab().unwrap(), &v).unwrap()] {
            for prefix in [b"".as_slice(),b"(",b"()",b"()(" ] {
                let mut a = c.start(); let mut r = reference.start();
                a.commit_bytes(prefix).unwrap(); r.commit_bytes(prefix).unwrap();
                assert_eq!(a.mask(), r.mask(), "{mode:?} after {prefix:?}");
                assert_eq!(a.is_accepting(), r.is_accepting());
                assert_eq!(a.forced(), r.forced());
                assert_eq!(allowed(&a.mask(),100), a.is_accepting());
                assert!(!allowed(&a.mask(),25));
                let snapshot = a.clone();
                a.commit_token(100).unwrap();
                if r.is_accepting() {
                    assert!(a.is_terminated()); assert!(!a.is_rejected());
                } else { assert!(a.is_rejected()); assert!(!a.is_terminated()); }
                assert!(a.mask().iter().all(|&w|w==0));
                a = snapshot;
                assert_eq!(a.mask(), r.mask(), "clone restores prior state");
                let actual_reject = a.commit_token(20);
                let expected_reject = r.commit_token(20);
                assert_eq!(actual_reject.is_err(), expected_reject.is_err());
                assert!(a.is_rejected()); assert!(!a.is_terminated());
            }
        }
    }
}

#[test]
fn force_and_exact_token_aliases_agree_across_modes() {
    let v = vocab();
    let g = Grammar::from_ebnf(r#"start ::= "(" ")""#);
    let reference = g.compile(&v).unwrap();
    for mode in MODES {
        let c = g.compile_with(&v, BuildOptions::default().optimization(mode)).unwrap();
        let mut a = c.start(); let mut r = reference.start();
        assert_eq!(a.forced(),r.forced());
        assert_eq!(a.mask(),r.mask());
        a.commit_token(15).unwrap(); r.commit_token(15).unwrap();
        assert_eq!(a.forced(),r.forced()); assert_eq!(a.mask(),r.mask());
        a.commit_token(2).unwrap(); assert!(a.is_accepting());
    }
    let g = Grammar::from_glrm("glrm 1; start start; extern token MARK; nt start = MARK;")
        .bind("MARK", v.token(15).unwrap()).unwrap();
    for mode in MODES {
        let c = g.compile_with(&v, BuildOptions::default().optimization(mode)).unwrap();
        let mut a = c.start(); assert!(allowed(&a.mask(),15)); assert!(!allowed(&a.mask(),0));
        a.commit_token(15).unwrap(); assert!(a.is_accepting());
    }
}

#[test]
fn source_dynamic_composition_and_precompiled_link_have_distinct_scopes() {
    let v = vocab();
    let child = Grammar::from_ebnf(r#"start ::= ")""#);
    let grammar = Grammar::from_glrm("glrm 1; start start; extern grammar child; nt start = \"(\" child;")
        .bind("child", &child).unwrap();
    let reference = grammar.compile(&v).unwrap();
    for mode in MODES {
        let c=grammar.compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();
        compare_prefixes(&c,&reference,&[0,2,7,15,20],3);
    }
    let module = grammar.compile_unlinked(&v).unwrap();
    for mode in MODES {
        let c=module.link_with(BuildOptions::default().optimization(mode)).unwrap();
        compare_prefixes(&c,&reference,&[0,2,7,15,20],3);
    }
}

fn compare_prefixes(actual:&Constraint, expected:&Constraint, ids:&[u32], depth:usize) {
    let mut pending=vec![Vec::new()];
    while let Some(prefix)=pending.pop() {
        let mut a=actual.start();let mut b=expected.start();
        for &id in &prefix { a.commit_token(id).unwrap();b.commit_token(id).unwrap(); }
        assert_eq!(a.mask(),b.mask(),"after {prefix:?}");
        assert_eq!(a.is_accepting(),b.is_accepting());
        assert_eq!(a.is_rejected(),b.is_rejected());
        if prefix.len()<depth && !b.is_rejected() {
            for &id in ids { if allowed(&b.mask(),id) { let mut next=prefix.clone();next.push(id);pending.push(next); } }
        }
    }
}

#[test]
fn mixed_frontends_and_nested_exact_identity_match_native_composition() {
    let v=Vocab::new_with_exact_token_ids(vec![(0,b"x".to_vec()),(1,b"a".to_vec()),
        (2,b"b".to_vec()),(3,b"y".to_vec()),(4,b"xaby".to_vec()),(5,b"ab".to_vec()),
        (6,b" ".to_vec()),(7,b"\t".to_vec()),(8,b"x ab y".to_vec()),(9,b"xab\ty".to_vec()),
        (10,b"\"a\"".to_vec()),(11,b"x\"a\"y".to_vec()),(33,b"a".to_vec())],[99]);
    let parent=Grammar::from_glrm("glrm 1; start root; extern grammar child; nt root = \"x\" child \"y\";");
    let children=[Grammar::from_ebnf(r#"start ::= "a" "b"?"#),
        Grammar::from_lark("start: \"a\" \"b\"?\n%ignore /\\t+/"),
        Grammar::from_glrm("glrm 1; start root; pragma glrmask { lexer group letters = A, B; } t A = \"a\"; t B = \"b\"; nt root = A B?;"),
        Grammar::from_json_schema(r#"{"const":"a"}"#)];
    for child in children {
        let description=parent.clone().bind("child",child).unwrap();
        let reference=description.compile(&v).unwrap();
        let actual=description.compile_with(&v,BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
        for c in [actual.clone(),Constraint::load(actual.save()).unwrap(),
            Constraint::load_with_vocab(actual.save_without_vocab().unwrap(),&v).unwrap()] {
            compare_prefixes(&c,&reference,&[0,1,2,3,4,5,6,7,8,9,10,11,33,99],5);
        }
    }
    let exact=Grammar::from_glrm("glrm 1; start root; extern token MARK; nt root = MARK;")
        .bind("MARK",v.tokens([33,99]).unwrap()).unwrap();
    let middle=Grammar::from_glrm("glrm 1; start root; extern grammar leaf; nt root = leaf;")
        .bind("leaf",exact).unwrap();
    let description=parent.bind("child",middle).unwrap();
    let reference=description.compile(&v).unwrap();
    let actual=description.compile_with(&v,BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
    compare_prefixes(&actual,&reference,&[0,1,3,4,8,33,99],4);
    assert!(!allowed(&actual.start().mask(),4));
}

#[test]
fn compressed_public_lark_uses_executable_lr_and_roundtrips() {
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"aa".to_vec())]);
    // start + r0..r30 reaches the established 32-rule compression threshold.
    let mut source=String::from("start: r0\n");
    for i in 0..30 { source.push_str(&format!("r{i}: \"a\" r{}\n",i+1)); }
    source.push_str("r30: \"b\"\n");
    let g=Grammar::from_lark(&source);
    let oracle=g.compile_with(&v,BuildOptions::default().optimization(Optimization::Balanced).end_tokens([100])).unwrap();
    let actual=g.compile_with(&v,BuildOptions::default().optimization(Optimization::FastBuild).end_tokens([100])).unwrap();
    assert_eq!(actual.start().mask()[0],5);
    for c in [actual.clone(),Constraint::load(actual.save()).unwrap(),
        Constraint::load_with_vocab(actual.save_without_vocab().unwrap(),&v).unwrap()] {
        for n in [0,1,15,29,30] {
            let prefix=vec![b'a';n]; let mut a=c.start();let mut o=oracle.start();
            a.commit_bytes(&prefix).unwrap();o.commit_bytes(&prefix).unwrap();
            assert_eq!(a.mask(),o.mask(),"after {n} a bytes");
            assert_eq!(a.is_accepting(),o.is_accepting());assert_eq!(a.forced(),o.forced());
            let snapshot=a.clone();
            if n==30 { a.commit_token(1).unwrap();assert!(a.is_accepting());
                assert!(allowed(&a.mask(),100));a.commit_token(100).unwrap();assert!(a.is_terminated()); }
            assert_eq!(snapshot.mask(),o.mask());
        }
        for n in [29,31] {
            let mut state=c.start();let mut invalid=vec![b'a';n];invalid.push(b'b');
            assert!(state.commit_bytes(&invalid).is_err());assert!(state.is_rejected());
        }
    }
}

#[test]
fn whole_word_tokens_remain_suggested_after_right_linear_compression() {
    for n in [29,30,63] {
        let mut source=String::from("start: r0\n");
        for i in 0..n {source.push_str(&format!("r{i}: \"a\" r{}\n",i+1));}
        source.push_str(&format!("r{n}: \"b\"\n"));
        let word=|count:usize| {let mut bytes=vec![b'a';count];bytes.push(b'b');bytes};
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"aa".to_vec()),
            (3,word(n)),(7,word(n)),(8,word(n-1)),(9,word(n+1)),(11,b"x".to_vec())]);
        let g=Grammar::from_lark(&source);
        let reference=g.compile_with(&v,BuildOptions::default().optimization(Optimization::FastBuild).end_tokens([100])).unwrap();
        for mode in MODES {
            let compiled=g.compile_with(&v,BuildOptions::default().optimization(mode).end_tokens([100])).unwrap();
            for c in [compiled.clone(),Constraint::load(compiled.save()).unwrap(),
                Constraint::load_with_vocab(compiled.save_without_vocab().unwrap(),&v).unwrap()] {
                for prefix in [Vec::new(),vec![b'a'],vec![b'a';n-1],vec![b'a';n],word(n)] {
                    let mut a=c.start();let mut expected=reference.start();
                    a.commit_bytes(&prefix).unwrap();expected.commit_bytes(&prefix).unwrap();
                    assert_eq!(a.mask(),expected.mask(),"{mode:?}, length {n}, prefix {prefix:?}");
                    assert_eq!(a.is_accepting(),expected.is_accepting());
                    assert_eq!(a.forced(),expected.forced());
                    let snapshot=a.clone();assert_eq!(snapshot.mask(),expected.mask());
                }
                let mut a=c.start();
                assert!(allowed(&a.mask(),3));assert!(allowed(&a.mask(),7));
                assert!(!allowed(&a.mask(),8));assert!(!allowed(&a.mask(),9));
                a.commit_token(7).unwrap();assert!(a.is_accepting());
                assert!(allowed(&a.mask(),100));a.commit_token(100).unwrap();assert!(a.is_terminated());
                for invalid in [8,9,11] {
                    let mut a=c.start();let mut expected=reference.start();
                    assert_eq!(a.commit_token(invalid).is_err(),expected.commit_token(invalid).is_err());
                    assert!(a.is_rejected());assert!(!a.is_terminated());
                }
            }
        }
    }
    // A missing suggestion can empty the entire mask with a one-token model.
    let mut source=String::from("start: r0\n");
    for i in 0..30 {source.push_str(&format!("r{i}: \"a\" r{}\n",i+1));}
    source.push_str("r30: \"b\"\n");let mut word=vec![b'a';30];word.push(b'b');
    let v=Vocab::new(vec![(3,word)]);
    for mode in MODES {
        let c=Grammar::from_lark(&source).compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();
        let mut a=c.start();assert_eq!(a.mask(),vec![8]);a.commit_token(3).unwrap();assert!(a.is_accepting());
    }
}
