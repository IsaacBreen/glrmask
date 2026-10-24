use glrmask::test_modes::MaskMode;
use glrmask::{Constraint,Grammar,Module,PreparedGrammar,TokenSet,Vocab};
const MODES: [MaskMode; 6] = MaskMode::ALL;
fn allowed(mask:&[u32],id:u32)->bool{mask.get(id as usize/32).is_some_and(|w|w&(1<<(id%32))!=0)}
fn accepts(c:&Constraint,ids:&[u32])->bool {let mut s=c.start();for &id in ids {if !allowed(&s.mask(),id) || !s.commit_token(id){return false;}}s.is_accepting()}
fn words(words:&[&str])->Vocab{Vocab::new(words.iter().enumerate().map(|(i,w)|(i as u32,w.as_bytes().to_vec())).collect())}
fn lark(source:&str)->Grammar{Grammar::from_lark(source).unwrap()}
#[test]
fn numeric_sets_complements_and_wildcards_use_actual_id_domain() {
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"a".to_vec()),(3,vec![])]).with_special_tokens([5,7]);
    for (source,expected) in [("start: <[1,3,5-5]>",vec![1,3,5]),("start: <[^0,3]>",vec![1,5,7]),("start: <[*]>",vec![0,1,3,5,7])]{
        for mode in MODES{
            let c=lark(source).compile_with_build_options(&v,mode.build_options().end_tokens([10])).unwrap();
            assert_eq!((0..=10).filter(|&id|allowed(&c.start().mask(),id)).collect::<Vec<_>>(),expected,"{source} {mode:?}");
            for &id in &expected{assert!(accepts(&c,&[id,10]));}
            let loaded=Constraint::load(c.save(), &v).unwrap();assert_eq!(loaded.start().mask(),c.start().mask());
        }
    }
    assert!(lark("start: <[0-5]>").compile(&v).is_err());
    // Exact-token atoms are ordinary Lark syntax in the single dialect.
    assert!(Grammar::from_lark("start: <[1]>").unwrap().compile(&v).is_ok());
}
#[test]
fn empty_event_set_is_not_epsilon() {
    let v=words(&["x","xx"]);
    for mode in MODES {
        let c=lark("start: <[]> | \"x\"").compile_with_build_options(&v,mode.build_options()).unwrap();
        assert!(!c.start().is_accepting());assert!(accepts(&c,&[0]));assert!(!accepts(&c,&[1]));
        let c=lark("start: <[^0-1]> | \"x\"").compile_with_build_options(&v,mode.build_options()).unwrap();
        assert!(!c.start().is_accepting());assert!(accepts(&c,&[0]));
    }
}
#[test]
fn named_tokens_are_token_externs_and_preserve_alias_identity() {
    let v=Vocab::new(vec![(0,b"<|end|>".to_vec()),(1,b"<|end|>".to_vec())]).with_special_tokens([4]);
    let g=lark("start: <|end|>");
    assert!(g.compile(&v).is_err());
    let externs=g.externs();
    assert_eq!(externs.len(),1);
    assert_eq!(externs.get("end").unwrap().kind(),glrmask::ExternKind::Token);
    for mode in MODES {
        let mut m=Module::new(&g);
        m.bind("end",TokenSet::new([1,4])).unwrap();
        let c=m.compile_with(&v,mode.compile_options()).unwrap();
        assert!(!allowed(&c.start().mask(),0));assert!(allowed(&c.start().mask(),1));assert!(allowed(&c.start().mask(),4));
    }
}
#[test]
fn imported_grammars_share_binding_api_and_do_not_collide_with_rules() {
    let v=words(&["x","y","xy"]);
    let g=lark("start: payload @payload\npayload: \"x\"");
    assert_eq!(g.requirements().unwrap()[0].name(),"payload");
    let widened=g.bind_snapshot("payload",TokenSet::from(0)).unwrap();
    for mode in MODES {
        let c=widened.compile_with_build_options(&v,mode.build_options()).unwrap();
        assert!(accepts(&c,&[0,0]));assert!(!accepts(&c,&[2]));
    }
    assert_eq!(g.requirements().unwrap().len(),1);
    let child=Grammar::from_glrm("start root nt root=\"y\"").unwrap();
    let p=g.prepare(&v).unwrap().bind_snapshot("payload",child.prepare(&v).unwrap()).unwrap();
    let loaded=PreparedGrammar::load(p.save(), &v).unwrap();
    for mode in MODES {
        let c=g.bind_snapshot("payload",&child).unwrap().compile_with_build_options(&v,mode.build_options()).unwrap();
        assert!(accepts(&c,&[2]));assert!(accepts(&loaded.link_with(mode.build_options()).unwrap(),&[2]));
    }
}
#[test]
fn inline_json_is_equivalent_to_host_bound_isolated_schema() {
    let source=r#"%ignore /[ ]+/
start: "<" %json {"type":"string","const":"a b"} ">""#;
    let schema=r#"{"type":"string","const":"a b"}"#;
    let v=words(&["<",">","\"a b\"","<\"a b\">","< \"a b\" >","\"ab\"","<\"ab\">"]);
    let host=lark("%ignore /[ ]+/\nstart: \"<\" @payload \">\"").bind_snapshot("payload",Grammar::from_json_schema(schema).unwrap()).unwrap();
    for mode in MODES{
        let options=mode.build_options();
        let a=lark(source).compile_with_build_options(&v,options.clone()).unwrap();let b=host.compile_with_build_options(&v,options).unwrap();
        assert_eq!(a.start().mask(),b.start().mask());
        assert!(accepts(&a,&[3]));assert!(!accepts(&a,&[6]));
    }
}
#[test]
fn lazy_regular_body_and_casefold_flags_have_exact_effects() {
    let v=words(&["a!","aa!","ABC","abc","a\nb"]);
    for mode in MODES{
        let c=lark("start: word \"!\"\nword[lazy]: /a+/").compile_with_build_options(&v,mode.build_options()).unwrap();
        assert!(accepts(&c,&[0]));assert!(!accepts(&c,&[1]));
        let c=Grammar::from_lark("start: \"abc\"i | /a.b/s").unwrap().compile_with_build_options(&v,mode.build_options()).unwrap();
        for id in [2,3,4]{assert!(accepts(&c,&[id]));}
    }
    assert!(Grammar::from_lark("start[lazy]: /a+/").unwrap().compile(&v).is_ok());
    assert!(Grammar::from_lark("start: /a+?/").is_err());
}

#[test]
fn ignored_terminals_remain_distinct_under_maximal_munch() {
    let v=words(&["ab","abb","b","a","<ab>","<abb>"]);
    let native=Grammar::from_glrm("ignore A ignore AB start root t A=\"a\" t AB=\"ab\" nt root=\"b\"").unwrap();
    let lark=Grammar::from_lark("%ignore \"a\"\n%ignore \"ab\"\nstart: \"b\"").unwrap();
    let union=Grammar::from_glrm("ignore ALL start root t ALL=\"a\" | \"ab\" nt root=\"b\"").unwrap();
    let nested=Grammar::from_glrm("start root grammar child{ignore A ignore AB start x t A=\"a\" t AB=\"ab\" nt x=\"b\"} nt root=\"<\" child \">\"").unwrap();
    for mode in MODES {
        let opts=mode.build_options();
        let a=native.compile_with_build_options(&v,opts.clone()).unwrap();let b=lark.compile_with_build_options(&v,opts.clone()).unwrap();
        for id in [0,1,2]{assert!(accepts(&a,&[id]),"native {mode:?} id{id}");assert!(accepts(&b,&[id]),"Lark {mode:?} id{id}");}
        assert!(!accepts(&union.compile_with_build_options(&v,opts.clone()).unwrap(),&[0]),"union must not be silently substituted");
        let c=nested.compile_with_build_options(&v,opts).unwrap();for id in [4,5]{assert!(accepts(&c,&[id]));}
    }
    for bad in ["ignore A ignore EMPTY start r t A=\"a\" t EMPTY=/b*/ nt r=\"c\"",
                "ignore A ignore X start r t A=\"a\" nt X=\"b\" nt r=\"c\""]{
        assert!(Grammar::from_glrm(bad).and_then(|g|g.compile(&v).map(|_|())).is_err());
    }
    assert!(Grammar::from_lark("%ignore \"a\"\n%ignore /b*/\nstart: \"b\"").unwrap().compile(&v).is_err());
}
