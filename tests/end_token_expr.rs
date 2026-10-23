use glrmask::{
    CompileOptions, Expr, Grammar, MaskStrategy, Module, TokenSet, Vocab,
};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"{".to_vec()),
        (1, b"}".to_vec()),
        (2, b" ".to_vec()),
    ])
    .with_special_tokens([99])
}

fn body() -> Grammar {
    Grammar::from_glrm(
        r#"
            glrm 1;
            start root;
            ignore WS;
            t WS = " "+;
            nt root = "{" "}";
        "#,
    )
}

fn root(grammar: Grammar, ids: impl IntoIterator<Item = u32>) -> Grammar {
    Grammar::from_expr(Expr::sequence([
        Expr::from(grammar),
        Expr::from(TokenSet::new(ids)),
    ]))
    .expect("root expression")
}

fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32)
        .is_some_and(|word| word & (1 << (id % 32)) != 0)
}

#[test]
fn end_token_expression_fast_path_is_exact() {
    for strategy in [MaskStrategy::ParserDwa, MaskStrategy::TokenTrie] {
        let v = vocab();
        let c = root(body(), [99])
            .compile_with(&v, CompileOptions::default().mask_strategy(strategy))
            .unwrap();
        assert!(!c.has_child_components());

        let mut state = c.start();
        assert!(!allowed(&state.mask(), 99));
        assert!(state.commit_token(0)); // "{"
        assert!(!allowed(&state.mask(), 99));
        assert!(state.commit_token(2)); // whitespace before the close
        assert!(state.commit_token(1)); // "}"
        assert!(allowed(&state.mask(), 99));

        let mut ws = c.start();
        assert!(ws.commit_token(0));
        assert!(ws.commit_token(1));
        assert!(ws.commit_token(2)); // trailing whitespace before eos
        assert!(ws.commit_token(99));
        assert!(ws.mask().iter().all(|word| *word == 0));
        assert!(!ws.commit_token(0));
        assert!(!ws.commit_bytes(b" "));
    }
}

#[test]
fn out_of_vocab_end_token_is_rejected() {
    let v = vocab();
    assert!(root(body(), [1000]).compile(&v).is_err());
}

#[test]
fn ignored_root_takes_the_normal_path_and_still_compiles() {
    let v = vocab();
    let g = Grammar::from_glrm(
        r#"
            glrm 1;
            start root;
            ignore WS;
            t WS = " "+;
            extern grammar child;
            extern token eos;
            nt root = child eos;
        "#,
    );
    let mut m = Module::new(&g);
    m.bind("child", body()).unwrap();
    m.bind("eos", TokenSet::new([99])).unwrap();
    let c = m.compile(&v).unwrap();
    assert!(c.has_child_components());
}

#[test]
fn bound_expression_over_precompiled_body_uses_the_fast_path() {
    let v = vocab();
    let g = Grammar::from_glrm("glrm 1; start root; extern grammar x; nt root = x;");
    let mut m = Module::new(&g);
    let precompiled = Module::new(&body());
    m.bind(
        "x",
        Expr::sequence([
            Expr::from(precompiled),
            Expr::from(TokenSet::new([99])),
        ]),
    )
    .unwrap();
    let c = m.compile(&v).unwrap();
    assert!(!c.has_child_components());
    let mut state = c.start();
    assert!(state.commit_token(0));
    assert!(state.commit_token(1));
    assert!(allowed(&state.mask(), 99));
    assert!(state.commit_token(99));
}

#[test]
fn empty_end_token_set_is_the_empty_language() {
    let v = vocab();
    match root(body(), std::iter::empty()).compile(&v) {
        Ok(c) => {
            let state = c.start();
            assert!(!state.is_accepting());
            assert!(state.mask().iter().all(|word| *word == 0));
        }
        Err(_) => {}
    }
}
