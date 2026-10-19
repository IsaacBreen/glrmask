//! Public-consumer lifecycle tests: no internal-api shortcuts or private field access.
use glrmask::{CompileOptions, Expr, Grammar, IntoExpr, Module, ModuleOptions, Optimization, Token, Vocab};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec()),
        (3, b",".to_vec()), (4, b"a,b".to_vec()), (5, Vec::new()),
    ])
}

#[test]
fn construction_and_options_do_not_change_the_source_grammar() {
    let source = Grammar::from_ebnf("start ::= \"a\"");
    let original = source.module();
    let different = original.with_options(
        ModuleOptions::default().optimization(Optimization::Balanced),
    ).unwrap();
    assert_eq!(source.module().options().optimization_value(), Optimization::Auto);
    assert_eq!(original.options().optimization_value(), Optimization::Auto);
    assert_eq!(different.options().optimization_value(), Optimization::Balanced);
    let v = vocab();
    assert_eq!(source.compile(&v).unwrap().start().mask(), different.compile(&v).unwrap().start().mask());
}

#[test]
fn prepared_parent_accepts_late_cold_children_and_preserves_snapshots() {
    let v = vocab();
    let source = Grammar::from_glrm("extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT");
    let mut prepared = source.module();
    prepared.precompile(&v).unwrap();
    let saved_open = prepared.save();
    let mut first = Module::load_with_vocab(&saved_open, &v).unwrap();
    let mut second = first.clone();
    first.bind("LEFT", Grammar::from_ebnf("start ::= \"a\"")).unwrap();
    first.bind("RIGHT", Grammar::from_ebnf("start ::= \"b\"")).unwrap();
    second.bind("LEFT", Grammar::from_ebnf("start ::= \"b\"")).unwrap();
    second.bind("RIGHT", Grammar::from_ebnf("start ::= \"a\"")).unwrap();
    assert_eq!(prepared.unbound().unwrap().len(), 2);
    assert_eq!(saved_open, prepared.save());
    assert_ne!(first.compile(&v).unwrap().start().mask(), second.compile(&v).unwrap().start().mask());
    let root = first.compile(&v).unwrap();
    let mut state = root.start();
    state.commit_token(2).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn native_local_modes_survive_expression_grouping_and_persistence() {
    let v = vocab();
    for a_mode in [Optimization::Auto, Optimization::Balanced, Optimization::FastRuntime] {
        for b_mode in [Optimization::Auto, Optimization::Balanced, Optimization::FastRuntime] {
            let a = Grammar::from_ebnf("start ::= \"a\"").module()
                .with_options(ModuleOptions::default().optimization(a_mode)).unwrap();
            let mut b = Grammar::from_ebnf("start ::= \"b\"").module()
                .with_options(ModuleOptions::default().optimization(b_mode)).unwrap();
            b.precompile(&v).unwrap();
            let expr = Expr::sequence([a.into_expr(), Expr::sequence([b]).into_expr()]);
            assert!(Grammar::from_expr(&expr).is_err(), "configured work cannot become raw source");
            let module = Module::from_expr(&expr).unwrap();
            let loaded = Module::load_with_vocab(module.save(), &v).unwrap();
            let root = loaded.compile(&v).unwrap();
            let mut state = root.start();
            assert_ne!(state.mask()[0] & (1 << 2), 0, "a model token may span both leaves");
            state.commit_token(2).unwrap();
            assert!(state.is_accepting());
            assert!(!state.is_rejected());
        }
    }
}

#[test]
fn anonymous_expression_paths_do_not_depend_on_operator_nesting() {
    let a = Grammar::from_glrm("extern grammar VALUE start s nt s=VALUE");
    let b = Grammar::from_ebnf("start ::= \",\"");
    let nested = Expr::sequence([Expr::sequence([&a]).into_expr(), b.into_expr(), a.into_expr()]);
    let mut module = Module::from_expr(&nested).unwrap();
    let names: Vec<_> = module.unbound().unwrap().iter().map(|x| x.name()).collect();
    assert_eq!(names, ["leaf0::VALUE", "leaf2::VALUE"]);
    let before = module.save();
    assert!(module.bind("leaf1::VALUE", Token::new(5)).is_err());
    assert_eq!(module.save(), before, "failed descendant binding must be atomic");
    module.bind("leaf0::VALUE", Grammar::from_ebnf("start ::= \"a\"")).unwrap();
    module.bind("leaf2::VALUE", Grammar::from_ebnf("start ::= \"b\"")).unwrap();
    assert!(module.unbound().unwrap().is_empty());
    assert_eq!(module.externs().unwrap().iter().filter(|x| x.name().ends_with("::VALUE")).count(), 2);
    let v = vocab();
    let root = module.compile(&v).unwrap();
    let mut state = root.start();
    state.commit_token(4).unwrap();
    assert!(state.is_accepting());
}

#[test]
fn changing_locally_frozen_options_is_atomic_but_cold_parent_is_independent() {
    let v = vocab();
    let mut leaf = Grammar::from_ebnf("start ::= \"a\"").module();
    leaf.precompile(&v).unwrap();
    let bytes = leaf.save();
    let changed = ModuleOptions::default().optimization(Optimization::Balanced);
    assert!(leaf.set_options(changed.clone()).is_err());
    assert_eq!(leaf.save(), bytes);
    let mut parent = Grammar::from_glrm("extern grammar VALUE start s nt s=VALUE").module();
    parent.bind("VALUE", &leaf).unwrap();
    parent.set_options(changed).unwrap();
    let root = parent.compile_with(&v, CompileOptions::default().end_tokens([5])).unwrap();
    let mut state = root.start();
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());
    state.commit_token(5).unwrap();
    assert!(state.is_accepting());
    assert!(!state.is_rejected());
}
