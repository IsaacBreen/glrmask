//! Public-consumer lifecycle tests: no internal-api shortcuts or private field access.
use glrmask::{BoundaryMaskStrategy, CompileOptions, Expr, Grammar, MaskStrategy, Module, Token, Vocab};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec()),
        (3, b",".to_vec()), (4, b"a,b".to_vec()), (5, Vec::new()),
    ])
}

#[test]
fn construction_and_options_do_not_change_the_source_grammar() {
    let source = Grammar::from_ebnf("start ::= \"a\"");
    let original = Module::new(&source);
    let mut different = original.clone();
    different.set_mask_strategy(MaskStrategy::TokenClassTrie).unwrap();
    assert_eq!(original.mask_strategy(), MaskStrategy::Auto);
    assert_eq!(different.mask_strategy(), MaskStrategy::TokenClassTrie);
    let v = vocab();
    assert_eq!(source.compile(&v).unwrap().start().mask(), different.compile(&v).unwrap().start().mask());
}

#[test]
fn prepared_parent_accepts_late_cold_children_and_preserves_snapshots() {
    let v = vocab();
    let source = Grammar::from_glrm("extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT");
    let mut prepared = Module::new(&source);
    prepared.precompile(&v).unwrap();
    let saved_open = prepared.save();
    let mut first = Module::load(&saved_open, Some(&v)).unwrap();
    let mut second = first.clone();
    first.bind("LEFT", Grammar::from_ebnf("start ::= \"a\"")).unwrap();
    first.bind("RIGHT", Grammar::from_ebnf("start ::= \"b\"")).unwrap();
    second.bind("LEFT", Grammar::from_ebnf("start ::= \"b\"")).unwrap();
    second.bind("RIGHT", Grammar::from_ebnf("start ::= \"a\"")).unwrap();
    assert_eq!(prepared.externs().unbound().len(), 2);
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
    for a_mode in [MaskStrategy::Auto, MaskStrategy::ParserDwa, MaskStrategy::TokenClassTrie] {
        for b_mode in [MaskStrategy::Auto, MaskStrategy::ParserDwa, MaskStrategy::TokenClassTrie] {
            let mut a = Module::new(&Grammar::from_ebnf("start ::= \"a\""));
            a.set_mask_strategy(a_mode).unwrap();
            let mut b = Module::new(&Grammar::from_ebnf("start ::= \"b\""));
            b.set_mask_strategy(b_mode).unwrap();
            b.precompile(&v).unwrap();
            let expr = Expr::sequence([Expr::from(a), Expr::sequence([Expr::from(b)])]);
            let mut host = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;"));
            host.bind("E", expr).unwrap();
            let loaded = Module::load(&host.save(), Some(&v)).unwrap();
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
fn open_child_and_qualified_paths_are_strictly_rejected() {
    let a = Grammar::from_glrm("extern grammar VALUE start s nt s=VALUE;");
    let mut parent = Module::new(&Grammar::from_glrm("extern grammar CHILD start s nt s=CHILD;"));
    let err = parent.bind("CHILD", Module::new(&a)).unwrap_err().to_string();
    assert!(err.contains("unbound externs"), "must reject open child: {err}");
    assert!(err.contains("VALUE"), "must list unbound externs: {err}");

    let err_qual = parent.bind("CHILD::VALUE", Token::new(5)).unwrap_err().to_string();
    assert!(err_qual.contains("qualified paths are not supported"), "must reject '::': {err_qual}");
}

#[test]
fn settings_setters_never_invalidate_retained_work() {
    let v = vocab();
    let mut leaf = Module::new(&Grammar::from_ebnf("start ::= \"a\""));
    leaf.precompile(&v).unwrap();
    let _before_save = leaf.save();

    // Setters store the value and never touch retained work
    leaf.set_mask_strategy(MaskStrategy::ParserDwa).unwrap();
    assert_eq!(leaf.mask_strategy(), MaskStrategy::ParserDwa);
    leaf.set_boundary_mask_strategy(BoundaryMaskStrategy::FastRuntime).unwrap();
    assert_eq!(leaf.boundary_mask_strategy(), BoundaryMaskStrategy::FastRuntime);

    let mut parent = Module::new(&Grammar::from_glrm("extern grammar VALUE start s nt s=VALUE;"));
    parent.bind("VALUE", &leaf).unwrap();
    parent.set_mask_strategy(MaskStrategy::ParserDwa).unwrap();
    let root = parent.compile_with(&v, CompileOptions::default().end_tokens([5])).unwrap();
    let mut state = root.start();
    state.commit_token(0).unwrap();
    assert!(state.is_accepting());
    state.commit_token(5).unwrap();
    assert!(state.is_accepting());
    assert!(!state.is_rejected());
}
