//! Public-consumer lifecycle tests: no internal-api shortcuts or private field access.
use glrmask::test_modes::MaskMode;
use glrmask::{BoundaryMaskStrategy, CompileOptions, Expr, Grammar, MaskStrategy, Module, TokenSet, Vocab};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec()),
        (3, b",".to_vec()), (4, b"a,b".to_vec()), (5, Vec::new()),
    ])
}

#[test]
fn construction_and_compile_options_do_not_change_the_source_grammar() {
    let source = Grammar::from_ebnf("start ::= \"a\"").unwrap();
    let module = Module::new(&source);
    let v = vocab();
    let default_mask = module.compile(&v).unwrap().start().mask();
    let explicit_mask = module
        .compile_with(&v, glrmask::test_modes::MaskMode::C.compile_options())
        .unwrap()
        .start()
        .mask();
    assert_eq!(default_mask, explicit_mask);
    assert_eq!(source.compile(&v).unwrap().start().mask(), default_mask);
}

#[test]
fn prepared_parent_accepts_late_cold_children_and_preserves_snapshots() {
    let v = vocab();
    let source = Grammar::from_glrm("extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT").unwrap();
    let mut prepared = Module::new(&source);
    prepared.precompile(&v).unwrap();
    let saved_open = prepared.save();
    let mut first = Module::load(&saved_open).unwrap();
    let mut second = first.clone();
    first.bind("LEFT", Grammar::from_ebnf("start ::= \"a\"").unwrap()).unwrap();
    first.bind("RIGHT", Grammar::from_ebnf("start ::= \"b\"").unwrap()).unwrap();
    second.bind("LEFT", Grammar::from_ebnf("start ::= \"b\"").unwrap()).unwrap();
    second.bind("RIGHT", Grammar::from_ebnf("start ::= \"a\"").unwrap()).unwrap();
    assert_eq!(prepared.externs().unbound().len(), 2);
    assert_eq!(saved_open, prepared.save());
    assert_ne!(first.compile(&v).unwrap().start().mask(), second.compile(&v).unwrap().start().mask());
    let root = first.compile(&v).unwrap();
    let mut state = root.start();
    assert!(state.commit_token(2));
    assert!(state.is_accepting());
}

#[test]
fn native_local_modes_survive_expression_grouping_and_persistence() {
    let v = vocab();
    for mode in MaskMode::ALL {
        let a = Module::new(&Grammar::from_ebnf("start ::= \"a\"").unwrap());
        let mut b = Module::new(&Grammar::from_ebnf("start ::= \"b\"").unwrap());
        b.precompile_with(&v, mode.compile_options()).unwrap();
        let expr = Expr::sequence([Expr::from(a), Expr::sequence([Expr::from(b)])]);
        let mut host = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;").unwrap());
        host.bind("E", expr).unwrap();
        let loaded = Module::load(&host.save()).unwrap();
        let root = loaded
            .compile_with(&v, mode.compile_options())
            .unwrap();
        let mut state = root.start();
        assert_ne!(state.mask()[0] & (1 << 2), 0, "a model token may span both leaves");
        assert!(state.commit_token(2));
        assert!(state.is_accepting());
    }
}

#[test]
fn open_child_and_qualified_paths_are_strictly_rejected() {
    let a = Grammar::from_glrm("extern grammar VALUE start s nt s=VALUE;").unwrap();
    let mut parent = Module::new(&Grammar::from_glrm("extern grammar CHILD start s nt s=CHILD;").unwrap());
    let err = parent.bind("CHILD", Module::new(&a)).unwrap_err().to_string();
    assert!(err.contains("unbound externs"), "must reject open child: {err}");
    assert!(err.contains("VALUE"), "must list unbound externs: {err}");

    let err_qual = parent.bind("CHILD::VALUE", TokenSet::from(5)).unwrap_err().to_string();
    assert!(err_qual.contains("qualified paths are not supported"), "must reject '::': {err_qual}");
}

#[test]
fn compile_options_never_invalidate_retained_work() {
    let v = vocab();
    let mut leaf = Module::new(&Grammar::from_ebnf("start ::= \"a\"").unwrap());
    leaf.precompile(&v).unwrap();
    let _before_save = leaf.save();

    let mut parent = Module::new(&Grammar::from_glrm("extern grammar VALUE start s nt s=VALUE;").unwrap());
    parent.bind("VALUE", &leaf).unwrap();
    let root = Grammar::from_expr(Expr::sequence([
        Expr::from(parent),
        Expr::from(TokenSet::new([5])),
    ]))
    .unwrap()
    .compile_with(
        &v,
        CompileOptions::default()
            .mask_strategy(MaskStrategy::ParserDwa)
            .boundary_mask_strategy(BoundaryMaskStrategy::ParserDwa),
    )
    .unwrap();
    let mut state = root.start();
    assert!(state.commit_token(0));
    // Grammar complete, but acceptance needs the configured end token.
    assert!(!state.is_accepting());
    assert!(state.commit_token(5));
    assert!(state.is_accepting());
}
