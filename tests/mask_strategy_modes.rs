//! The public mask-strategy surface: the A/B/C TokenTrie configurations and
//! the rejected combination D.
use glrmask::test_modes::MaskMode;
use glrmask::{CompileOptions, ErrorKind, Grammar, MaskStrategy, Module, ParserBackend, Vocab};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"(".to_vec()),
        (1, b")".to_vec()),
        (2, b"()".to_vec()),
        (3, b"x".to_vec()),
        (4, b"a".to_vec()),
        (5, b"b".to_vec()),
        (6, b"ab".to_vec()),
    ])
}

fn d_options() -> CompileOptions {
    CompileOptions::default()
        .mask_strategy(MaskStrategy::TokenTrie)
        .merge_equivalent_tokens(true)
}

fn assert_d_rejected(error: glrmask::Error) {
    match error.kind() {
        ErrorKind::StrategyUnsupported(message) => assert_eq!(
            message, "merging equivalent tokens requires template DFAs",
        ),
        other => panic!("expected StrategyUnsupported, got {other:?}"),
    }
}

#[test]
fn merging_equivalent_tokens_without_template_dfas_is_rejected_at_every_entry_point() {
    let vocab = vocab();
    let grammar = Grammar::from_ebnf(r#"start ::= "a""#).unwrap();
    let module = Module::new(&grammar);

    assert_d_rejected(grammar.compile_with(&vocab, d_options()).unwrap_err());
    assert_d_rejected(grammar.precompile_with(&vocab, d_options()).unwrap_err());
    assert_d_rejected(module.compile_with(&vocab, d_options()).unwrap_err());
    assert_d_rejected(module.precompile_with(&vocab, d_options()).unwrap_err());

    // The check is about the pair of switches, not the requested strategy.
    let auto_d = d_options().mask_strategy(MaskStrategy::Auto);
    assert_d_rejected(grammar.compile_with(&vocab, auto_d).unwrap_err());
}

fn assert_same_masks(a: &glrmask::Constraint, b: &glrmask::Constraint) {
    assert_eq!(a.start().mask(), b.start().mask(), "initial masks differ");
    for id in 0..8u32 {
        let mut left = a.start();
        let mut right = b.start();
        assert_eq!(left.commit_token(id), right.commit_token(id), "commit {id}");
        assert_eq!(left.mask(), right.mask(), "masks differ after {id}");
        assert_eq!(left.is_accepting(), right.is_accepting(), "acceptance after {id}");
    }
}

#[test]
fn template_dfas_is_table_free_and_the_default_token_trie_keeps_its_table() {
    let vocab = vocab();
    let source = r#"start ::= "(" start ")" start | """#;

    let grammar = Grammar::from_ebnf(source).unwrap();
    let a = grammar
        .compile_with(&vocab, MaskMode::A.compile_options())
        .unwrap();
    let b = grammar
        .compile_with(&vocab, MaskMode::B.compile_options())
        .unwrap();
    assert_eq!(a.parser_backend(), ParserBackend::LrTable, "A must keep the LR table");
    assert_eq!(b.parser_backend(), ParserBackend::TemplateDfa, "B must be table-free");
    assert_same_masks(&a, &b);

    // The same two configurations over a module with a bound child.
    let child = Grammar::from_ebnf(r#"start ::= "a""#).unwrap();
    let host = || {
        let mut host = Module::new(
            &Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "(" child ")" | "x";"#,
            )
            .unwrap(),
        );
        host.bind("child", child.clone()).unwrap();
        host
    };
    let a = host()
        .compile_with(&vocab, MaskMode::A.compile_options())
        .unwrap();
    let b = host()
        .compile_with(&vocab, MaskMode::B.compile_options())
        .unwrap();
    assert_eq!(a.parser_backend(), ParserBackend::LrTable, "composed A must keep the LR table");
    assert_eq!(b.parser_backend(), ParserBackend::TemplateDfa, "composed B must be table-free");
    assert_same_masks(&a, &b);
}
