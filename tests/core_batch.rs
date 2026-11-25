use glrmask::test_modes::MaskMode;
use glrmask::{BuildOptions, Expr, Grammar, InternalMaskStrategy, Module, PreparedGrammar, TokenSet, Vocab};
fn options()->BuildOptions {BuildOptions::default().optimization(InternalMaskStrategy::FAST_COMPILE)}
fn vocab()->Vocab {Vocab::new(vec![(0,b"x".to_vec()),(1,b"y".to_vec())]).unwrap()}

#[test]
fn single_binding_mutates_returns_unit_and_failure_is_atomic() {
    let mut g=Module::new(&Grammar::from_glrm("extern token MARK start root nt root=MARK").unwrap());
    let untouched=g.clone();
    let result:()=g.bind("MARK",TokenSet::new([17])).unwrap();assert_eq!(result,());
    assert!(g.externs().unbound().is_empty());assert_eq!(untouched.externs().unbound().len(),1);
    assert!(g.bind("MARK",TokenSet::new([29])).is_err());
    let c=g.compile_with(&vocab(),glrmask::CompileOptions::default().internal_mask_strategy(InternalMaskStrategy::FAST_COMPILE)).unwrap();let mask=c.start().mask();
    assert_ne!(mask[0]&(1<<17),0);assert_eq!(mask[0]&(1<<29),0);
}
#[test]
fn module_bind_accepts_every_binding_kind() {
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(7,b"x".to_vec()),(8,b"y".to_vec()),(9,b"z".to_vec())]).unwrap();
    let owned=Grammar::from_glrm("start root nt root=\"a\"").unwrap();
    let borrowed=Grammar::from_glrm("start root nt root=\"b\"").unwrap();
    let tokens=TokenSet::new([7]);
    let borrowed_tokens=TokenSet::new([8]);
    let mut g=Grammar::from_glrm(
        "extern grammar OWNED extern grammar BORROWED extern grammar EXPR extern token TOKENS extern token SET extern token ID start root nt root=OWNED BORROWED EXPR TOKENS SET ID"
    ).unwrap();
    g.bind("OWNED",owned).unwrap();
    g.bind("BORROWED",&borrowed).unwrap();
    g.bind("EXPR",Expr::literal("a")).unwrap();
    let mut g=Module::new(&g);
    g.bind("TOKENS",tokens).unwrap();
    g.bind("SET",&borrowed_tokens).unwrap();
    g.bind("ID",glrmask::TokenSet::new([9u32])).unwrap();
    assert!(g.externs().unbound().is_empty());
    let c=g.compile_with(&v,glrmask::CompileOptions::default().internal_mask_strategy(InternalMaskStrategy::FAST_COMPILE)).unwrap();let mut s=c.start();
    for id in [0,1,0,7,8,9] {assert!(s.commit_token(id));}
    assert!(s.is_accepting());
}

#[test]
fn source_and_prepared_children_are_snapshots_not_live_handles() {
    let v=vocab();let text="extern token MARK start root nt root=MARK";
    let parent_text="extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT";
    let mut child=Grammar::from_glrm(text).unwrap();
    let mut parent=Grammar::from_glrm(parent_text).unwrap();parent.bind("LEFT",&child).unwrap();parent.bind("RIGHT",&child).unwrap();
    child=child.bind_snapshot("MARK",TokenSet::new([29])).unwrap();
    assert!(child.requirements().unwrap().is_empty());
    assert_eq!(parent.requirements().unwrap().len(),2);
    let mut open=parent.prepare(&v).unwrap();
    let old=open.clone();open.bind("LEFT::MARK",TokenSet::new([17])).unwrap();
    assert_eq!(open.requirements().unwrap()[0].name(),"RIGHT::MARK");
    assert_eq!(old.requirements().unwrap().len(),2);
    let mut loaded=PreparedGrammar::load(open.save(), &v).unwrap();
    loaded.bind("RIGHT::MARK",TokenSet::new([29])).unwrap();
    let c=loaded.link_with(options()).unwrap();let mut s=c.start();
    assert_ne!(s.mask()[0]&(1<<17),0);assert!(s.commit_token(17));
    assert_ne!(s.mask()[0]&(1<<29),0);assert!(s.commit_token(29));assert!(s.is_accepting());
    assert_eq!(open.requirements().unwrap().len(),1);
}
#[test]
fn event_values_widen_to_grammar_but_grammars_do_not_narrow_to_token_ports() {
    let v=vocab();
    for mode in MaskMode::ALL {
        let opt=mode.build_options();
        let mut a=Module::new(&Grammar::from_glrm("extern grammar VALUE start root nt root=VALUE").unwrap());
        a.bind("VALUE",TokenSet::new([17,29])).unwrap();
        let mut b=Grammar::from_glrm("extern grammar VALUE start root nt root=VALUE").unwrap().prepare(&v).unwrap();
        b.bind("VALUE",TokenSet::new([17,29])).unwrap();
        let x=a.compile_with(&v,mode.compile_options()).unwrap();let y=b.link_with(opt).unwrap();
        assert_eq!(x.start().mask(),y.start().mask(),"{mode:?}");
        for c in [&x,&y] {
            let mut s=c.start();assert!(!s.commit_bytes(b"x"));
            let mut s=c.start();assert!(s.commit_token(17));assert!(s.is_accepting());
        }
    }
    let mut t=Grammar::from_glrm("extern token VALUE start root nt root=VALUE").unwrap();
    assert!(t.bind("VALUE",Grammar::from_glrm("start r nt r=\"x\"").unwrap()).is_err());
    assert_eq!(t.requirements().unwrap().len(),1);
}
#[test]
fn independent_concurrent_specializations_do_not_share_mutations() {
    let v=vocab();
    let p=Grammar::from_glrm("extern token MARK start root nt root=MARK").unwrap().prepare(&v).unwrap();
    std::thread::scope(|scope| {
        let jobs=(0..8).map(|i| {let p=&p;scope.spawn(move || {
            let mut request=p.clone();let id=if i%2==0{17}else{29};
            request.bind("MARK",TokenSet::new([id])).unwrap();
            let c=request.link_with(options()).unwrap();let mut s=c.start();
            assert!(s.commit_token(id));assert!(s.is_accepting());
        })}).collect::<Vec<_>>();
        for job in jobs {job.join().unwrap();}
    });
    assert_eq!(p.requirements().unwrap().len(),1);
}
