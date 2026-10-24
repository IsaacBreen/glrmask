use glrmask::test_modes::MaskMode;
use glrmask::{Constraint, ExternKind, Grammar, PreparedGrammar, TokenSet, Vocab};
const MODES: [MaskMode; 5] = MaskMode::ALL;
const LEAF:&str="extern token MARK start body nt body=\"x\" MARK";
const PAIR:&str="extern grammar LEFT extern grammar RIGHT start root nt root=LEFT \"|\" RIGHT";
fn vocab()->Vocab {Vocab::new(vec![(0,b"x".to_vec()),(1,b"|".to_vec()),(2,b"x".to_vec())]).with_special_tokens([4,5])}
fn allowed(mask:&[u32],id:u32)->bool {mask.get(id as usize/32).is_some_and(|w|w&(1<<(id%32))!=0)}
fn check_pair(c:&Constraint) {
    let mut s=c.start();
    for (id,next) in [(0,vec![4]),(4,vec![1]),(1,vec![0,2]),(2,vec![5])] {
        assert!(allowed(&s.mask(),id),"missing next event {id}");assert!(s.commit_token(id));
        let mask=s.mask();let actual=(0..6).filter(|&i|allowed(&mask,i)).collect::<Vec<_>>();assert_eq!(actual,next,"after {id}");
    }
    assert!(s.commit_token(5));assert!(s.is_accepting());
}
#[test]
fn typed_source_ports_reject_wrong_kind_at_binding() {
    let token=Grammar::from_glrm(LEAF).unwrap();let grammar=Grammar::from_glrm(PAIR).unwrap();
    assert!(token.bind_snapshot("MARK",Grammar::from_glrm("start root nt root=\"x\"").unwrap()).is_err());
    let widened=grammar.bind_snapshot("LEFT",TokenSet::from(4)).unwrap();
    assert_eq!(widened.requirements().unwrap().iter().map(|r|r.name()).collect::<Vec<_>>(),["RIGHT"]);
    assert_eq!(grammar.requirements().unwrap().len(),2);
    let requirements=token.requirements().unwrap();assert_eq!(requirements[0].kind,ExternKind::Token);
    assert_eq!(requirements[0].name(),"MARK");
    assert!(token.bind_snapshot("MARK",TokenSet::new([4])).is_ok());
}
#[test]
fn source_open_attachments_have_independent_qualified_environments() {
    let v=vocab();let leaf=Grammar::from_glrm(LEAF).unwrap();
    let pair=Grammar::from_glrm(PAIR).unwrap().bind_snapshot("LEFT",&leaf).unwrap().bind_snapshot("RIGHT",&leaf).unwrap();
    assert_eq!(pair.requirements().unwrap().iter().map(|r|r.name()).collect::<Vec<_>>(),["LEFT::MARK","RIGHT::MARK"]);
    let complete=pair.bind_snapshot("LEFT::MARK",TokenSet::from(4)).unwrap().bind_snapshot("RIGHT::MARK",TokenSet::from(5)).unwrap();
    assert!(complete.bind_snapshot("LEFT::MARK",TokenSet::from(5)).is_err());
    assert_eq!(leaf.requirements().unwrap().len(),1);
    for mode in MODES {check_pair(&complete.compile_with_build_options(&v,mode.build_options()).unwrap());}
}
#[test]
fn prepared_open_children_survive_serialization_and_independent_binding() {
    let v=vocab();let leaf=Grammar::from_glrm(LEAF).unwrap().prepare(&v).unwrap();
    let pair=Grammar::from_glrm(PAIR).unwrap().prepare(&v).unwrap().bind_snapshot("LEFT",&leaf).unwrap().bind_snapshot("RIGHT",&leaf).unwrap();
    assert!(pair.link().is_err());assert_eq!(pair.requirements().unwrap().len(),2);
    let loaded=PreparedGrammar::load(pair.save(),&v).unwrap();
    assert!(loaded.bind_snapshot("LEFT::MARK",Grammar::from_glrm("start x nt x=\"x\"").unwrap().prepare(&v).unwrap()).is_err());
    let complete=loaded.bind_snapshot("LEFT::MARK",TokenSet::from(4)).unwrap().bind_snapshot("RIGHT::MARK",TokenSet::from(5)).unwrap();
    for mode in MODES {check_pair(&complete.link_with(mode.build_options()).unwrap());}
    assert_eq!(pair.requirements().unwrap().len(),2);assert_eq!(leaf.requirements().unwrap().len(),1);
}
#[test]
fn explicit_exact_adapter_has_event_not_spelling_semantics() {
    let v=vocab();let event=Grammar::exact(TokenSet::new([2,4]));
    let host=Grammar::from_glrm("extern grammar VALUE start root nt root=VALUE").unwrap();
    for mode in MODES {
        let c=host.bind_snapshot("VALUE",&event).unwrap().compile_with_build_options(&v,mode.build_options()).unwrap();
        let mask=c.start().mask();assert!(!allowed(&mask,0));assert!(allowed(&mask,2));assert!(allowed(&mask,4));assert!(!allowed(&mask,5));
        for id in [2,4] {let mut s=c.start();assert!(s.commit_token(id));assert!(s.is_accepting());}
    }
    let other=Vocab::new(vec![(0,b"different".to_vec()),(1,b"|".to_vec()),(2,b"x".to_vec())]);
    assert!(event.prepare(&other).is_err(), "ID 4 is absent from the target vocabulary");
}
#[test]
fn unused_parameters_still_belong_to_the_interface() {
    let v=vocab();let g=Grammar::from_glrm("extern token UNUSED start root nt root=\"x\"").unwrap();
    assert!(g.compile(&v).is_err());
    let prepared=g.prepare(&v).unwrap();assert!(prepared.link().is_err());
    let c=prepared.bind_snapshot("UNUSED",TokenSet::from(4)).unwrap().link().unwrap();
    let mut s=c.start();assert!(s.commit_token(0));assert!(s.is_accepting());
}
#[test]
fn inline_and_attached_paths_have_the_same_scope_rules() {
    let source = "grammar inner {extern token MARK start x nt x=MARK} start root nt root=inner";
    let err = Grammar::from_glrm(source).unwrap_err().to_string();
    assert!(err.contains("extern declarations are only allowed at the top level"), "{err}");
}


#[test]
fn requirement_coordinates_survive_preparation_and_save_load() {
    let source="// α\nextern token MARK\nstart root\nnt root = MARK";
    let g=Grammar::from_glrm(source).unwrap();
    let r=g.requirements().unwrap().remove(0);
    let span=r.source_span.unwrap();
    assert_eq!(&source[span.start..span.end],"MARK");
    assert_eq!((span.line,span.column),(2,14));
    let v=vocab();let p=g.prepare(&v).unwrap();
    assert_eq!(p.requirements().unwrap(),[r.clone()]);
    let loaded=PreparedGrammar::load(p.save(),&v).unwrap();
    assert_eq!(loaded.requirements().unwrap(),[r]);
    let child=Grammar::from_glrm("start root nt root=\"x\"").unwrap();
    assert_eq!(g.bind_snapshot("MARK",&child).unwrap_err().source_span(),Some(span));
    assert_eq!(loaded.bind_snapshot("MARK",child.prepare(&v).unwrap()).unwrap_err().source_span(),Some(span));
}

#[test]
fn lark_requirements_and_cached_errors_retain_original_locations() {
    let source="// α\nstart: @payload @payload";
    let g=Grammar::from_lark(source).unwrap();
    let r=g.requirements().unwrap().remove(0);let span=r.source_span.unwrap();
    assert_eq!(&source[span.start..span.end],"@payload");
    assert_eq!((span.line,span.column),(2,8));
    let v=vocab();assert_eq!(g.prepare(&v).unwrap().requirements().unwrap(),[r]);
    let invalid="start root\nnt root = )";
    let span=Grammar::from_glrm(invalid).unwrap_err().source_span().unwrap();
    assert_eq!(&invalid[span.start..span.end],")");
    let unknown="start: <|unknown|>";
    let g=Grammar::from_lark(unknown).unwrap();
    let externs=g.externs();
    assert_eq!(externs.len(),1);
    let declared=externs.get("unknown").expect("a named atom declares a token extern");
    assert_eq!(declared.name(),"unknown");
    assert_eq!(declared.kind(),glrmask::ExternKind::Token);
    // A named atom stays open while preparing and must be bound before compile.
    assert_eq!(g.prepare(&v).unwrap().requirements().unwrap()[0].name(),"unknown");
    assert!(g.compile(&v).is_err());
}


#[test]
fn compact_closed_dag_does_not_expand_during_inspection_or_persistence() {
    let v=vocab();
    let leaf=Grammar::from_glrm("start root nt root=\"x\"").unwrap().prepare(&v).unwrap();
    let wrapper=Grammar::from_glrm("extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT").unwrap().prepare(&v).unwrap();
    let unit_bytes=leaf.save().len()+wrapper.save().len();
    let mut graph=leaf;
    // This compact object describes more than two billion invocation paths.
    // Introspection and persistence must operate on the shared instance DAG.
    for _ in 0..31 {graph=wrapper.bind_snapshot("LEFT",&graph).unwrap().bind_snapshot("RIGHT",&graph).unwrap();}
    assert!(graph.requirements().unwrap().is_empty());
    let bytes=graph.save();
    assert!(bytes.len()<unit_bytes*3+128_000,"artifact unexpectedly expanded: {} vs unit {unit_bytes}",bytes.len());
    let loaded=PreparedGrammar::load(&bytes,&v).unwrap();
    assert!(loaded.requirements().unwrap().is_empty());
    assert_eq!(bytes,loaded.save(),"DAG round trip is canonical and preserves sharing");
    assert!(loaded.link().unwrap_err().to_string().contains("resource limit"));
    let mut corrupt=bytes.clone();corrupt.extend_from_slice(b"trailing");
    assert!(PreparedGrammar::load(corrupt, &vocab()).is_err());
    for cut in [0,8,15,bytes.len()-1] {assert!(PreparedGrammar::load(&bytes[..cut], &vocab()).is_err());}
}

#[test]
fn compact_open_dag_has_bounded_requirements_but_remains_serializable() {
    let v=vocab();let leaf=Grammar::from_glrm(LEAF).unwrap().prepare(&v).unwrap();
    let wrapper=Grammar::from_glrm(PAIR).unwrap().prepare(&v).unwrap();let mut graph=leaf;
    for _ in 0..18 {graph=wrapper.bind_snapshot("LEFT",&graph).unwrap().bind_snapshot("RIGHT",&graph).unwrap();}
    assert!(graph.requirements().unwrap_err().to_string().contains("resource limit"));
    let bytes=graph.save();assert!(bytes.len()<2_000_000);
    let loaded=PreparedGrammar::load(bytes,&v).unwrap();
    assert!(loaded.requirements().is_err());assert!(loaded.link().is_err());
}

#[test]
fn source_preparation_memoizes_shared_attachments_and_checks_depth() {
    let v=vocab();
    let wrapper=Grammar::from_glrm("extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT").unwrap();
    let mut source=Grammar::from_glrm("start root nt root=\"x\"").unwrap();
    for _ in 0..18 {source=wrapper.bind_snapshot("LEFT",&source).unwrap().bind_snapshot("RIGHT",&source).unwrap();}
    assert!(source.requirements().unwrap().is_empty());
    let prepared=source.prepare(&v).unwrap();assert!(prepared.requirements().unwrap().is_empty());
    assert!(prepared.save().len()<8_000_000);
    let one=Grammar::from_glrm("extern grammar CHILD start root nt root=CHILD").unwrap().prepare(&v).unwrap();
    let mut graph=Grammar::from_glrm("start root nt root=eps").unwrap().prepare(&v).unwrap();
    for _ in 0..64 {graph=one.bind_snapshot("CHILD",&graph).unwrap();}
    assert!(one.bind_snapshot("CHILD",&graph).unwrap_err().to_string().contains("depth"));
    assert!(PreparedGrammar::load(graph.save(), &vocab()).is_ok());
}


#[test]
fn empty_exact_bindings_are_empty_languages_not_epsilon() {
    let v=vocab();let empty=TokenSet::new([]);
    assert_eq!(empty, TokenSet::new([]));
    assert_eq!(TokenSet::from_ranges(std::iter::empty::<std::ops::Range<u32>>()), empty);
    let parent=Grammar::from_glrm("extern token EVENT start root nt root=EVENT | \"x\"").unwrap();
    let source=parent.bind_snapshot("EVENT",empty.clone()).unwrap();
    let prepared=parent.prepare(&v).unwrap().bind_snapshot("EVENT",empty.clone()).unwrap();
    let loaded=PreparedGrammar::load(prepared.save(),&v).unwrap();
    let event=Grammar::exact(empty);
    for mode in MODES {
        let options=mode.build_options();
        for c in [source.compile_with_build_options(&v,options.clone()).unwrap(),loaded.link_with(options.clone()).unwrap()] {
            assert!(!c.start().is_accepting());assert_eq!(c.start().mask(),Grammar::from_glrm("start root nt root=\"x\"").unwrap().compile_with_build_options(&v,options.clone()).unwrap().start().mask());
        }
        let c=event.compile_with_build_options(&v,options).unwrap();
        assert!(!c.start().is_accepting());assert!(c.start().mask().iter().all(|word|*word==0));
        assert!(!Constraint::load(c.save(), &vocab()).unwrap().start().is_accepting());
    }
}

#[test]
fn public_native_parser_rejects_compatibility_only_forms() {
    for source in ["extern VALUE start root nt root=VALUE", "start root nt root ::= 'x';",
        "start root nt root=@token(4)","start root nt root=<[4]>","start root nt root=%json {}",
        "g child={start root nt root=\"x\"};start root nt root=child",
        "start root nt root=fa{start 0 accept 1 0 -> 1:eps}"] {
        assert!(Grammar::from_glrm(source).and_then(|g|g.requirements().map(|_|())).is_err(),"{source}");
    }
}
