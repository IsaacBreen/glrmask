use glrmask::{BuildOptions, Grammar, MaskStrategy, PreparedGrammar, TokenSet, Vocab};
fn options()->BuildOptions {BuildOptions::default().optimization(MaskStrategy::TokenTrie)}
fn vocab()->Vocab {Vocab::new(vec![(0,b"x".to_vec()),(1,b"y".to_vec())]).with_special_tokens([17,29])}

#[test]
fn single_binding_mutates_returns_unit_and_failure_is_atomic() {
    let mut g=Grammar::from_glrm("extern token MARK start root nt root=MARK");
    let untouched=g.clone();
    let result:()=g.bind("MARK",TokenSet::from(17)).unwrap();assert_eq!(result,());
    assert!(g.requirements().unwrap().is_empty());assert_eq!(untouched.requirements().unwrap().len(),1);
    assert!(g.bind("MARK",TokenSet::from(29)).is_err());
    let c=g.compile_with_build_options(&vocab(),options()).unwrap();let mask=c.start().mask();
    assert_ne!(mask[0]&(1<<17),0);assert_eq!(mask[0]&(1<<29),0);
}
#[test]
fn source_and_prepared_children_are_snapshots_not_live_handles() {
    let v=vocab();let text="extern token MARK start root nt root=MARK";
    let parent_text="extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT";
    let mut child=Grammar::from_glrm(text);
    let mut parent=Grammar::from_glrm(parent_text);parent.bind("LEFT",&child).unwrap();parent.bind("RIGHT",&child).unwrap();
    child.bind("MARK",TokenSet::from(29)).unwrap();
    assert_eq!(parent.requirements().unwrap().len(),2);
    let mut open=parent.prepare(&v).unwrap();
    let old=open.clone();open.bind("LEFT::MARK",TokenSet::from(17)).unwrap();
    assert_eq!(open.requirements().unwrap()[0].name(),"RIGHT::MARK");
    assert_eq!(old.requirements().unwrap().len(),2);
    let mut loaded=PreparedGrammar::load_with_vocab(open.save(),&v).unwrap();
    loaded.bind("RIGHT::MARK",TokenSet::from(29)).unwrap();
    let c=loaded.link_with(options()).unwrap();let mut s=c.start();
    assert_ne!(s.mask()[0]&(1<<17),0);s.commit_token(17).unwrap();
    assert_ne!(s.mask()[0]&(1<<29),0);s.commit_token(29).unwrap();assert!(s.is_accepting());
    assert_eq!(open.requirements().unwrap().len(),1);
}
#[test]
fn event_values_widen_to_grammar_but_grammars_do_not_narrow_to_token_ports() {
    let v=vocab();
    for mode in [MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa] {
        let opt=BuildOptions::default().optimization(mode);
        let mut a=Grammar::from_glrm("extern grammar VALUE start root nt root=VALUE");
        a.bind("VALUE",TokenSet::new([17,29])).unwrap();
        let mut b=Grammar::from_glrm("extern grammar VALUE start root nt root=VALUE").prepare(&v).unwrap();
        b.bind("VALUE",TokenSet::new([17,29])).unwrap();
        let x=a.compile_with_build_options(&v,opt.clone()).unwrap();let y=b.link_with(opt).unwrap();
        assert_eq!(x.start().mask(),y.start().mask(),"{mode:?}");
        for c in [&x,&y] {
            let mut s=c.start();assert!(s.commit_bytes(b"x").is_err());
            let mut s=c.start();s.commit_token(17).unwrap();assert!(s.is_accepting());
        }
    }
    let mut t=Grammar::from_glrm("extern token VALUE start root nt root=VALUE");
    assert!(t.bind("VALUE",Grammar::from_glrm("start r nt r=\"x\"")).is_err());
    assert_eq!(t.requirements().unwrap().len(),1);
    let mut g=Grammar::from_glrm("extern grammar VALUE start root nt root=VALUE");
    g.bind("VALUE",TokenSet::from(12345)).unwrap();
    assert!(g.compile_with_build_options(&v,options()).is_err());
}
#[test]
fn independent_concurrent_specializations_do_not_share_mutations() {
    let v=vocab();
    let p=Grammar::from_glrm("extern token MARK start root nt root=MARK").prepare(&v).unwrap();
    std::thread::scope(|scope| {
        let jobs=(0..8).map(|i| {let p=&p;scope.spawn(move || {
            let mut request=p.clone();let id=if i%2==0{17}else{29};
            request.bind("MARK",TokenSet::from(id)).unwrap();
            let c=request.link_with(options()).unwrap();let mut s=c.start();
            s.commit_token(id).unwrap();assert!(s.is_accepting());
        })}).collect::<Vec<_>>();
        for job in jobs {job.join().unwrap();}
    });
    assert_eq!(p.requirements().unwrap().len(),1);
}
