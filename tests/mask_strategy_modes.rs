//! The public mask-strategy surface: the A/B/C/D TokenTrie configurations.
use glrmask::test_modes::MaskMode;
use glrmask::{CompileOptions, Grammar, MaskStrategy, Module, ParserBackend, TokenTrie, Vocab};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"(".to_vec()),
        (1, b")".to_vec()),
        (2, b"()".to_vec()),
        (3, b"x".to_vec()),
        (4, b"a".to_vec()),
        (5, b"b".to_vec()),
        (6, b"ab".to_vec()),
        (7, b"y".to_vec()),
        (8, b"xy".to_vec()),
        (9, b"xay".to_vec()),
        (10, b"ay".to_vec()),
        (11, b"\"".to_vec()),
        (12, b"\"a\"".to_vec()),
        (13, b",".to_vec()),
        (14, b"[".to_vec()),
        (15, b"]".to_vec()),
        (16, Vec::new()),
    ])
}

fn d_options() -> CompileOptions {
    CompileOptions::default().mask_strategy(
        TokenTrie::new()
            .parser_backend(ParserBackend::LrTable)
            .merge_equivalent_tokens(true),
    )
}

fn a_options() -> CompileOptions {
    CompileOptions::default().mask_strategy(TokenTrie::new())
}

fn assert_same_masks(a: &glrmask::Constraint, b: &glrmask::Constraint) {
    assert_eq!(a.start().mask(), b.start().mask(), "initial masks differ");
    for id in 0..17u32 {
        let mut left = a.start();
        let mut right = b.start();
        assert_eq!(left.commit_token(id), right.commit_token(id), "commit {id}");
        assert_eq!(left.mask(), right.mask(), "masks differ after {id}");
        assert_eq!(left.is_accepting(), right.is_accepting(), "acceptance after {id}");
    }
}

fn compare_exhaustive(
    a: &glrmask::ConstraintState<'_>,
    d: &glrmask::ConstraintState<'_>,
    depth: usize,
    label: &str,
) {
    assert_eq!(a.mask(), d.mask(), "{label}: mask at depth {depth}");
    assert_eq!(a.is_accepting(), d.is_accepting(), "{label}: acceptance at depth {depth}");
    if depth == 0 {
        return;
    }
    for word_index in 0..a.mask().len() {
        let mut bits = a.mask()[word_index];
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let token = (word_index * 32 + bit as usize) as u32;
            bits &= bits - 1;
            let mut next_a = a.clone();
            let mut next_d = d.clone();
            assert_eq!(next_a.commit_token(token), next_d.commit_token(token), "{label}: commit {token}");
            compare_exhaustive(&next_a, &next_d, depth - 1, label);
        }
    }
}

struct Case {
    name: &'static str,
    grammar: Grammar,
    vocab: Vocab,
}

fn cases() -> Vec<Case> {
    let vocab = vocab();
    vec![
        Case {
            name: "ebnf",
            grammar: Grammar::from_ebnf(r#"start ::= "x" ["a"] "y" | "(" start ")""#).unwrap(),
            vocab: vocab.clone(),
        },
        Case {
            name: "glrm",
            grammar: Grammar::from_glrm(
                r#"glrm 1; start root; nt root = "x" "a"? "y" | "(" root ")" | "b";"#,
            )
            .unwrap(),
            vocab: vocab.clone(),
        },
        Case {
            name: "lark",
            grammar: Grammar::from_lark(r#"start: "x" ["a"] "y" | "(" start ")""#).unwrap(),
            vocab: vocab.clone(),
        },
        Case {
            name: "json-schema",
            grammar: Grammar::from_json_schema(
                r#"{"type":"array","items":{"enum":["\"a\"","\"\""]},"maxItems":2}"#,
            )
            .unwrap(),
            vocab: vocab.clone(),
        },
    ]
}

#[test]
fn d_is_lr_quotiented_and_matches_a_on_every_supported_frontend() {
    for case in cases() {
        let a = case.grammar.compile_with(&case.vocab, a_options()).unwrap();
        let d = case.grammar.compile_with(&case.vocab, d_options()).unwrap();
        assert_eq!(a.parser_backend(), ParserBackend::LrTable, "{} A", case.name);
        assert_eq!(d.parser_backend(), ParserBackend::LrTable, "{} D", case.name);
        assert!(d.runtime_has_lr_table(), "{} D must keep the LR table", case.name);
        assert!(!d.runtime_has_template_parser(), "{} D must not install a template parser", case.name);
        compare_exhaustive(&a.start(), &d.start(), 3, case.name);
        assert_same_masks(&a, &d);

        let loaded = glrmask::Constraint::load(d.save(), &case.vocab).unwrap();
        assert_eq!(loaded.parser_backend(), ParserBackend::LrTable, "{} D reload", case.name);
        assert!(loaded.runtime_has_lr_table(), "{} D reload table", case.name);
        assert!(!loaded.runtime_has_template_parser(), "{} D reload templates", case.name);
        compare_exhaustive(&d.start(), &loaded.start(), 2, case.name);
    }
}

#[test]
fn d_precompile_retains_nothing_and_saves_like_a_never_precompiled_module() {
    for case in cases() {
        for mode in [MaskMode::A, MaskMode::D] {
            let plain = Module::new(&case.grammar);
            let mut precompiled = Module::new(&case.grammar);
            precompiled
                .precompile_with(&case.vocab, mode.compile_options())
                .unwrap_or_else(|error| panic!("{} {mode:?} precompile: {error}", case.name));
            assert_eq!(
                precompiled.save(),
                plain.save(),
                "{} {mode:?}: an LR TokenTrie precompile must retain nothing",
                case.name,
            );
            let from_plain = plain.compile_with(&case.vocab, mode.compile_options()).unwrap();
            let from_pre = precompiled
                .compile_with(&case.vocab, mode.compile_options())
                .unwrap();
            compare_exhaustive(&from_plain.start(), &from_pre.start(), 2, case.name);
        }
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

#[test]
fn mask_strategy_accepts_token_trie_settings_directly() {
    let vocab = vocab();
    let grammar = Grammar::from_ebnf(r#"start ::= "a""#).unwrap();
    let d = TokenTrie::new().merge_equivalent_tokens(true);
    let compiled = grammar
        .compile_with(&vocab, CompileOptions::default().mask_strategy(d))
        .unwrap();
    assert_eq!(compiled.parser_backend(), ParserBackend::LrTable);
    assert!(compiled.runtime_is_grammar_quotiented());
    assert_eq!(MaskStrategy::FAST_COMPILE, MaskStrategy::TokenTrie(TokenTrie::new()));
}
