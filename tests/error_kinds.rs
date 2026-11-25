use glrmask::{ErrorKind, Expr, Grammar, Module, Vocab};

#[test]
fn syntax_and_source_span() {
    let e = Grammar::from_glrm("start s nt s = )").unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Syntax);
    assert!(e.source_span().is_some());
}

#[test]
fn caller_errors_are_invalid_input() {
    let mut grammar = Grammar::from_glrm("start s extern token E nt s = E").unwrap();
    assert_eq!(grammar.bind("missing", glrmask::TokenSet::new([0u32])).unwrap_err().kind(), ErrorKind::InvalidInput);
    let vocab = Vocab::new(vec![(0, b"a".to_vec())]).unwrap();
    assert_eq!(grammar.compile(&vocab).unwrap_err().kind(), ErrorKind::InvalidInput);
    grammar.bind("E", glrmask::TokenSet::new([0u32])).unwrap();
    assert_eq!(grammar.bind("E", glrmask::TokenSet::new([0u32])).unwrap_err().kind(), ErrorKind::InvalidInput);
}

#[test]
fn nested_expression_is_a_resource_limit() {
    let mut expr = Expr::literal("a");
    for _ in 0..70 { expr = Expr::sequence([expr]); }
    assert_eq!(Grammar::from_expr(expr).unwrap_err().kind(), ErrorKind::ResourceLimit);
}

#[test]
fn invalid_saved_bytes_are_serialization_error() {
    assert_eq!(Module::load(&[1, 2, 3]).unwrap_err().kind(), ErrorKind::Serialization);
}

#[test]
fn error_kind_is_copy_and_hash() {
    let kind = ErrorKind::Syntax;
    let set = std::collections::HashSet::from([kind]);
    assert!(set.contains(&kind));
}
