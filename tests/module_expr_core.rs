use glrmask::{CompileOptions, Expr, Grammar, IntoExpr, Module, ModuleOptions, Optimization, Token, Vocab};
fn vocab()->Vocab {Vocab::new_with_exact_token_ids(vec![(0,b"x".to_vec()),(1,b"a".to_vec()),(2,b"b".to_vec()),(3,b"y".to_vec()),(4,b"xay".to_vec()),(5,b"ab".to_vec()),(6,b"xby".to_vec()),(7,b"a".to_vec()),(8,b"aab".to_vec())],[99])}
fn allowed(mask:&[u32],id:u32)->bool {mask.get(id as usize/32).is_some_and(|w|w&(1<<(id%32))!=0)}
const HOST:&str="glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";";
const OPEN:&str="glrm 1; start start; extern grammar item; nt start = item;";
fn a()->Grammar<'static>{Grammar::from_ebnf("start ::= \"a\"")}
fn b()->Grammar<'static>{Grammar::from_ebnf("start ::= \"b\"")}
fn accepts(c:&glrmask::Constraint,ids:&[u32])->bool {let mut s=c.start();for &id in ids{if s.commit_token(id).is_err(){return false;}}s.is_accepting()}
#[test]
fn prepared_parent_raw_child_and_atomic_snapshots(){
    let v=vocab();let g=Grammar::from_glrm(HOST);let mut parent=g.module();parent.precompile(&v).unwrap();
    let empty=parent.clone();let mut child=Grammar::from_glrm(OPEN).module();
    parent.bind("child",&child).unwrap();child.bind("item",b()).unwrap();
    assert_eq!(parent.unbound().unwrap()[0].name(),"child::item");
    parent.bind("child::item",a()).unwrap();let bytes=parent.save();
    assert!(parent.bind("child::item",b()).is_err());assert_eq!(parent.save(),bytes);
    let c=parent.compile(&v).unwrap();assert!(accepts(&c,&[4]));assert!(!accepts(&c,&[6]));
    assert!(empty.compile(&v).is_err());assert_eq!(g.unbound().unwrap()[0].name(),"child");
    assert_eq!(parent.externs().unwrap().iter().map(|s|s.name()).collect::<Vec<_>>(),["child","child::item"]);
    assert!(parent.externs().unwrap()[0].source_span.is_some());
}
#[test]
fn options_are_local_and_retained_changes_are_atomic(){
    let v=vocab();let balanced=ModuleOptions::default().optimization(Optimization::Balanced);
    let mut child=a().module().with_options(balanced.clone()).unwrap();child.precompile(&v).unwrap();
    assert_eq!(child.options(),&balanced);let bytes=child.save();
    assert!(child.set_options(ModuleOptions::default()).is_err());assert_eq!(child.save(),bytes);
    assert!(child.with_options(balanced).is_ok());
    let mut parent=Grammar::from_glrm(HOST).module();parent.bind("child",&child).unwrap();
    let parent=parent.with_options(ModuleOptions::default().optimization(Optimization::FastRuntime)).unwrap();
    assert!(accepts(&parent.compile(&v).unwrap(),&[4]));
    assert_eq!(child.options(),&ModuleOptions::default().optimization(Optimization::Balanced));
}
#[test]
fn mixed_roundtrip_retains_open_raw_and_prepared_nodes(){
    let v=vocab();let mut parent=Grammar::from_glrm(HOST).module();parent.precompile(&v).unwrap();
    let mut child=Grammar::from_glrm(OPEN).module();child.bind("item",a()).unwrap();parent.bind("child",child).unwrap();
    let loaded=Module::load_with_vocab(parent.save(),&v).unwrap();assert!(accepts(&loaded.compile(&v).unwrap(),&[4]));
    assert!(loaded.with_options(ModuleOptions::default().optimization(Optimization::Balanced)).is_err());
    let raw=Module::load(a().module().save()).unwrap();assert!(accepts(&raw.compile(&v).unwrap(),&[1]));
    let mut open=Module::load(Grammar::from_glrm(OPEN).module().save()).unwrap();open.bind("item",a()).unwrap();assert!(accepts(&open.compile(&v).unwrap(),&[1]));
}
#[test]
fn expr_crosses_boundaries_and_parentheses_do_not_change_paths(){
    let v=vocab();let open=Grammar::from_glrm(OPEN);
    let left=Expr::sequence([open.into_expr(),b().into_expr()]);
    let expr=Expr::sequence([Expr::sequence([left])]);let mut m=Module::from_expr(&expr).unwrap();
    assert_eq!(m.unbound().unwrap()[0].name(),"leaf0::item");m.bind("leaf0::item",a()).unwrap();
    assert!(accepts(&m.compile(&v).unwrap(),&[5]));
    let union=Expr::choice([a(),b()]);let mut host=Grammar::from_glrm(HOST).module();host.bind("child",&union).unwrap();
    let c=host.compile(&v).unwrap();assert!(accepts(&c,&[4]));assert!(accepts(&c,&[6]));
    let source=Grammar::from_expr(&union).unwrap();assert!(accepts(&source.compile(&v).unwrap(),&[2]));
}
#[test]
fn expr_repeated_leaf_occurrences_remain_independent(){
    let v=vocab();let g=Grammar::from_glrm(OPEN);let mut m=Module::from_expr(&Expr::sequence([&g,&g])).unwrap();
    m.bind("leaf0::item",a()).unwrap();m.bind("leaf1::item",b()).unwrap();
    let mut loaded=Module::load(m.save()).unwrap();loaded.precompile(&v).unwrap();assert!(accepts(&loaded.compile(&v).unwrap(),&[5]));
    assert_eq!(g.unbound().unwrap().len(),1);
}
#[test]
fn configured_expr_leaves_cannot_be_silently_demoted_to_source(){
    let v=vocab();let mut leaf=a().module().with_options(ModuleOptions::default().optimization(Optimization::Balanced)).unwrap();
    leaf.precompile(&v).unwrap();let expr=Expr::sequence([leaf.into_expr(),b().into_expr()]);
    assert!(Grammar::from_expr(&expr).is_err());
    let module=Module::from_expr(&expr).unwrap();assert!(accepts(&module.compile(&v).unwrap(),&[5]));
}
#[test]
fn empty_operators_have_explicit_behavior(){
    let v=vocab();let epsilon=Module::from_expr(&Expr::sequence(Vec::<Expr>::new())).unwrap();assert!(epsilon.compile(&v).unwrap().start().is_accepting());
    assert!(Module::from_expr(&Expr::choice(Vec::<Expr>::new())).is_err());
}
#[test]
fn exact_events_identity_and_kind_widening_survive_storage(){
    let v=vocab();let host=Grammar::from_glrm("glrm 1; start start; extern token mark; nt start = mark;");
    let mut m=host.module();assert!(m.bind("mark",a()).is_err());m.bind("mark",v.tokens([7,99]).unwrap()).unwrap();
    let loaded=Module::load(m.save()).unwrap();let c=loaded.compile(&v).unwrap();let mask=c.start().mask();
    assert!(allowed(&mask,7));assert!(allowed(&mask,99));assert!(!allowed(&mask,1));
    let wrong=Vocab::new_with_exact_token_ids(vec![(7,b"different".to_vec())],[99]);assert!(Module::load_with_vocab(m.save(),&wrong).is_err());assert!(loaded.compile(&wrong).is_err());
    let mut wide=Grammar::from_glrm(OPEN).module();wide.bind("item",Token::new(7)).unwrap();let c=wide.compile(&v).unwrap();assert!(accepts(&c,&[7]));assert!(!allowed(&c.start().mask(),1));
}
#[test]
fn fastbuild_retains_representation_and_mixed_modes_fail_explicitly(){
    let v=vocab();let mut leaf=a().module().with_options(ModuleOptions::default().optimization(Optimization::FastBuild)).unwrap();leaf.precompile(&v).unwrap();
    let leaf=Module::load(leaf.save()).unwrap();assert!(accepts(&leaf.compile(&v).unwrap(),&[1]));
    let mut parent=Grammar::from_glrm(HOST).module();parent.bind("child",leaf).unwrap();
    let e=parent.compile(&v).unwrap_err().to_string();assert!(e.contains("FastBuild/LR"),"{e}");
}
#[test]
fn final_end_tokens_are_root_only(){
    let v=vocab();let m=a().module();let c=m.compile_with(&v,CompileOptions::default().end_tokens([99])).unwrap();let mut state=c.start();assert!(!allowed(&state.mask(),99));state.commit_token(1).unwrap();assert!(allowed(&state.mask(),99));state.commit_token(99).unwrap();
}
#[test]
fn malformed_storage_is_rejected(){
    let bytes=a().module().save();for len in [0,7,15,bytes.len()-1]{assert!(Module::load(&bytes[..len]).is_err());}
    let mut bad=bytes.clone();bad[7]=b'9';assert!(Module::load(bad).is_err());let mut bad=bytes;bad.push(0);assert!(Module::load(bad).is_err());
}

#[test]
fn complete_scoped_expression_masks_match_direct_source_reference(){
    use glrmask::{BuildOptions,LarkOptions};
    let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec()),(3,b"a b".to_vec()),(4,b" " .to_vec()),(5,b"aa".to_vec()),(6,b"aab".to_vec()),(7,b"c".to_vec()),(8,Vec::new())]);
    let first=Grammar::from_lark_with("start: \"a\"+\n%ignore /[ ]+/",LarkOptions::default());
    let second=Grammar::from_ebnf("start ::= \"b\" | \"\"");
    let expr=Expr::sequence([first.into_expr(),second.into_expr()]);
    let source=Grammar::from_expr(&expr).unwrap();
    let reference=source.compile_with(&v,BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
    for mode in [Optimization::Auto,Optimization::Balanced,Optimization::FastRuntime]{
        let mut module=Module::from_expr(&expr).unwrap().with_options(ModuleOptions::default().optimization(mode)).unwrap();
        module.precompile(&v).unwrap();let c=Module::load_with_vocab(module.save(),&v).unwrap().compile(&v).unwrap();
        let mut pending=vec![(c.start(),reference.start(),0)];
        while let Some((actual,expected,depth))=pending.pop(){
            assert_eq!(actual.mask(),expected.mask(),"{mode:?}, depth {depth}");
            assert_eq!(actual.is_accepting(),expected.is_accepting());assert_eq!(actual.is_rejected(),expected.is_rejected());
            assert_eq!(actual.forced_tokens().unwrap(),expected.forced_tokens().unwrap());
            if depth<3 {for id in 0..9 {let mut a=actual.clone();let mut e=expected.clone();
                assert_eq!(a.commit_token(id).is_ok(),e.commit_token(id).is_ok());
                if !a.is_rejected(){pending.push((a,e,depth+1));}
            }}
        }
    }
}
#[test]
fn exact_source_leaves_keep_target_identity_without_runtime_preparation(){
    let v=vocab();let grammar=Grammar::exact(v.tokens([7,99]).unwrap());let expr=Expr::sequence([grammar]);
    let module=Module::load(Module::from_expr(&expr).unwrap().save()).unwrap();
    assert!(accepts(&module.compile(&v).unwrap(),&[99]));let wrong=Vocab::new(vec![(7,b"a".to_vec())]);assert!(module.compile(&wrong).is_err());
}
#[test]
fn open_precompilation_and_binding_mismatch_are_atomic(){
    let v=vocab();let mut parent=Grammar::from_glrm(HOST).module();parent.precompile(&v).unwrap();let original=parent.save();
    let wrong=Vocab::new(vec![(0,b"other".to_vec())]);assert!(parent.precompile(&wrong).is_err());assert_eq!(parent.save(),original);
    assert!(parent.bind("child",wrong.tokens([0]).unwrap()).is_err());assert_eq!(parent.save(),original);
    let mut child=a().module();child.precompile(&wrong).unwrap();assert!(parent.bind("child",child).is_err());assert_eq!(parent.save(),original);
    let mut child=Grammar::from_glrm(OPEN).module();child.bind("item",Token::new(777)).unwrap();
    assert!(parent.bind("child",child).is_err());assert_eq!(parent.save(),original);
}
#[test]
fn raw_module_is_owned_and_can_preserve_deferred_source_errors(){
    let module={let text=String::from("start ::= \"a\"");Grammar::from_ebnf(&text).module()};
    assert!(accepts(&Module::load(module.save()).unwrap().compile(&vocab()).unwrap(),&[1]));
    let invalid=Grammar::from_glrm("invalid source").module();let loaded=Module::load(invalid.save()).unwrap();assert!(loaded.compile(&vocab()).is_err());
}
#[test]
fn compact_dags_stay_compact_until_bounded_runtime_construction(){
    let host=Grammar::from_glrm("glrm 1; start s; extern grammar left; extern grammar right; nt s = left | right;");
    let mut module=a().module();
    for _ in 0..19 {let mut parent=host.module();parent.bind("left",&module).unwrap();parent.bind("right",&module).unwrap();module=parent;}
    let bytes=module.save();assert!(bytes.len()<32_000,"{}",bytes.len());let loaded=Module::load(bytes).unwrap();
    assert!(loaded.unbound().unwrap().is_empty(),"closed DAG diagnostics need no occurrence expansion");
    assert!(loaded.externs().is_err(),"all-slot diagnostic expansion is bounded");
    assert!(loaded.compile(&vocab()).is_err(),"runtime expansion is bounded");
}
#[test]
fn local_modes_retain_actual_native_or_lr_storage_across_load(){
    for mode in [Optimization::Auto,Optimization::Balanced,Optimization::FastRuntime,Optimization::FastBuild]{
        let v=vocab();let mut module=a().module().with_options(ModuleOptions::default().optimization(mode)).unwrap();
        module.precompile(&v).unwrap();let loaded=Module::load_with_vocab(module.save(),&v).unwrap();
        assert_eq!(loaded.options(),module.options());
        let report=glrmask::__private::parser_backend_report(&loaded.compile(&v).unwrap());
        let text=report.to_string();
        assert_eq!(report["lr_table_present"].as_bool(),Some(mode==Optimization::FastBuild),"{mode:?}: {text}");
    }
}
#[test]
fn lark_inline_schema_preparation_preserves_native_and_dynamic_modes(){
    let v=Vocab::new(vec![(0,b"<".to_vec()),(1,b">".to_vec()),(2,b"\"a b\"".to_vec()),(3,b"<\"a b\">".to_vec()),(4,b"<\"ab\">".to_vec())]);
    let g=Grammar::from_lark_with(r#"start: "<" %json {"type":"string","const":"a b"} ">""#,glrmask::LarkOptions::llm());
    for mode in [Optimization::Auto,Optimization::Balanced,Optimization::FastRuntime,Optimization::FastBuild]{
        let mut m=g.module().with_options(ModuleOptions::default().optimization(mode)).unwrap();m.precompile(&v).unwrap();
        let c=Module::load(m.save()).unwrap().compile(&v).unwrap();assert!(accepts(&c,&[3]),"{mode:?}");assert!(!allowed(&c.start().mask(),4));
    }
}
#[test]
fn grammar_binds_source_expressions_directly_and_rejects_configured_leaves_atomically(){
    let v=vocab();let expr=Expr::choice([a(),b()]);
    let mut source=Grammar::from_glrm(HOST);source.bind("child",&expr).unwrap();let c=source.compile(&v).unwrap();assert!(accepts(&c,&[4]));assert!(accepts(&c,&[6]));
    let mut source=Grammar::from_glrm(HOST);source.bind("child",expr).unwrap();assert!(accepts(&source.compile(&v).unwrap(),&[4]));
    let configured=a().module().with_options(ModuleOptions::default().optimization(Optimization::Balanced)).unwrap();
    let mut source=Grammar::from_glrm(HOST);assert!(source.bind("child",configured.into_expr()).is_err());assert_eq!(source.unbound().unwrap()[0].name(),"child");
}
