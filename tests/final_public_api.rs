use glrmask::{
    BoundaryMaskStrategy, CompileOptions, Constraint, Expr, ExternKind, Grammar, MaskStrategy,
    Module, TokenSet, Vocab,
};

fn vocab() -> Vocab {
    Vocab::new(vec![
        (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
        (3, b"y".to_vec()), (4, b"xay".to_vec()), (5, b"ab".to_vec()),
        (7, b"a".to_vec()), (8, Vec::new()),
    ])
}

fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|w| w & (1 << (id % 32)) != 0)
}

const HOST: &str = "glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";";
const TOKEN_HOST: &str = "glrm 1; start start; extern token MARK; nt start = \"x\" MARK \"y\";";
const OPEN: &str = "glrm 1; start start; extern grammar item; nt start = item;";

#[test]
fn immutable_source_and_compiled_bindings_cross_token_boundaries() {
    let v = vocab();
    let parent = Grammar::from_glrm(HOST);
    let child = Grammar::from_ebnf(r#"start ::= "a""#);

    let mut m1 = Module::new(&parent);
    m1.bind("child", Module::new(&child)).unwrap();
    let c1 = m1.compile(&v).unwrap();

    let mut m2 = Module::new(&parent);
    let mut cchild = Module::new(&child);
    cchild.precompile(&v).unwrap();
    m2.precompile(&v).unwrap();
    m2.bind("child", cchild).unwrap();
    let c2 = m2.compile(&v).unwrap();

    for c in [c1, c2] {
        let mut s = c.start();
        assert!(allowed(&s.mask(), 4));
        assert!(s.commit_token(4));
        assert!(s.is_accepting());
    }
    assert!(Module::new(&parent).compile(&v).is_err());
}

#[test]
fn module_reuse_load_and_closed_link() {
    let v = vocab();
    let mut host = Module::new(&Grammar::from_glrm(HOST));
    host.precompile(&v).unwrap();
    let mut a = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    a.precompile(&v).unwrap();
    let mut b = Module::new(&Grammar::from_ebnf(r#"start ::= "b""#));
    b.precompile(&v).unwrap();

    for parent in [&host, &Module::load(&host.save()).unwrap()] {
        assert!(parent.compile(&v).is_err());
        let mut bound_a = parent.clone();
        bound_a.bind("child", &a).unwrap();
        let ca = bound_a.compile(&v).unwrap();
        assert!(allowed(&ca.start().mask(), 4));

        let mut bound_b = parent.clone();
        bound_b.bind("child", &b).unwrap();
        let cb = bound_b.compile(&v).unwrap();
        assert!(!allowed(&cb.start().mask(), 4));
        assert!(parent.compile(&v).is_err());
    }
}

#[test]
fn source_exact_token_identity_is_not_bytes() {
    let v = vocab();
    let mut m = Module::new(&Grammar::from_glrm(TOKEN_HOST));
    m.bind("MARK", TokenSet::from(7)).unwrap();
    let c = m.compile(&v).unwrap();
    let mut s = c.start();
    assert!(!allowed(&s.mask(), 4));
    assert!(s.commit_token(0));
    assert!(allowed(&s.mask(), 7));
    assert!(!allowed(&s.mask(), 1));
    assert!(s.commit_token(7));
    assert!(s.commit_token(3));
    assert!(s.is_accepting());
}

#[test]
fn compiled_exact_token_bind_matches_source_at_every_step() {
    let v = vocab();
    let mut m_expected = Module::new(&Grammar::from_glrm(TOKEN_HOST));
    m_expected.bind("MARK", TokenSet::new([7, 8])).unwrap();
    let expected = m_expected.compile(&v).unwrap();

    let mut m = Module::new(&Grammar::from_glrm(TOKEN_HOST));
    m.precompile(&v).unwrap();
    assert!(m.compile(&v).is_err());
    m.bind("MARK", TokenSet::new([7, 8])).unwrap();
    let actual = m.compile(&v).unwrap();

    let mut a = actual.start();
    let mut b = expected.start();
    for id in [0, 7, 3] {
        assert_eq!(a.mask(), b.mask(), "before {id}");
        assert!(a.commit_token(id));
        assert!(b.commit_token(id));
        assert_eq!(a.is_accepting(), b.is_accepting());
    }
    assert_eq!(a.mask(), b.mask());
}

#[test]
fn unbound_exact_token_survives_save_load() {
    let v = vocab();
    let mut m = Module::new(&Grammar::from_glrm(TOKEN_HOST));
    m.precompile(&v).unwrap();
    let mut loaded = Module::load(&m.save()).unwrap();
    assert!(loaded.compile(&v).is_err(), "loading must not silently close unresolved token slots");
    loaded.bind("MARK", TokenSet::from(7)).unwrap();
    let c = loaded.compile(&v).unwrap();
    let mut s = c.start();
    assert!(s.commit_token(0));
    assert!(allowed(&s.mask(), 7));
    assert!(!allowed(&s.mask(), 1));
}

#[test]
fn token_membership_and_compiled_vocabulary_are_validated() {
    let v = vocab();
    let other = Vocab::new(vec![(7, b"different".to_vec())]);
    let mut parent = Module::new(&Grammar::from_glrm(TOKEN_HOST));
    parent.precompile(&v).unwrap();
    assert!(parent.bind("MARK", TokenSet::from(6)).is_err());

    let mut widened = Module::new(&Grammar::from_glrm(HOST));
    widened.bind("child", TokenSet::from(7)).unwrap();
    let c = widened.compile(&v).unwrap();
    let mut state = c.start();
    assert!(state.commit_token(0));
    assert!(allowed(&state.mask(), 7));
    assert!(!allowed(&state.mask(), 1));
    assert!(state.commit_token(7));
    assert!(state.commit_token(3));
    assert!(state.is_accepting());

    let mut child = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    child.precompile(&v).unwrap();
    assert!(parent.bind("MARK", &child).is_err());

    let mut incompatible = Module::new(&Grammar::from_ebnf(r#"start ::= "different""#));
    incompatible.precompile(&other).unwrap();
    let mut host_m = Module::new(&Grammar::from_glrm(HOST));
    host_m.precompile(&v).unwrap();
    assert!(host_m.bind("child", &incompatible).is_err());
}

#[test]
fn artifact_kinds_and_corrupt_module_headers_are_rejected() {
    let v = vocab();
    let mut module = Module::new(&Grammar::from_glrm(TOKEN_HOST));
    module.precompile(&v).unwrap();
    let bytes = module.save();
    assert!(Constraint::load(bytes.clone(), &vocab()).is_err());
    assert!(bytes.starts_with(b"GLRASM03"));

    let constraint = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    assert!(Module::load(&constraint.save()).is_err());
    for length in [0, 1, 8, 15, 16, bytes.len() - 1] {
        assert!(Module::load(&bytes[..length]).is_err(), "accepted truncation to {length}");
    }
    let mut corrupt = bytes.clone();
    corrupt[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Module::load(&corrupt).is_err());
}

#[test]
fn root_end_tokens_can_extend_the_mask_without_changing_body_caches() {
    let v = vocab();
    let module = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    let root = Grammar::from_expr(Expr::sequence([
        Expr::from(module.clone()),
        Expr::from(TokenSet::new([130, 255])),
    ])).unwrap();
    let mut constraint = root.compile(&v).unwrap();
    for _ in 0..4 {
        assert_eq!(constraint.mask_len(), 8);
        let mut state = constraint.start();
        let mut buffer = vec![u32::MAX; 10];
        state.fill_mask(&mut buffer);
        assert_eq!(&buffer[..8], state.mask());
        assert_eq!(&buffer[8..], &[0, 0]);
        assert!(!allowed(&state.mask(), 130));
        assert!(state.commit_token(1));
        // Grammar complete, but acceptance needs an end token.
        assert!(!state.is_accepting());
        assert!(allowed(&state.mask(), 130));
        assert!(allowed(&state.mask(), 255));
        let mut other = state.clone();
        assert!(state.commit_token(130));
        assert!(state.is_accepting());
        assert_eq!(state.mask(), vec![0; 8]);
        assert!(state.forced_canonical_reference().is_empty());
        assert!(!state.commit_token(1));
        assert!(!state.commit_bytes(b"a"));
        assert!(!other.is_accepting());
        assert!(allowed(&other.mask(), 130));
        assert!(allowed(&other.mask(), 255));
        assert!(other.commit_token(255));
        assert!(other.is_accepting());
        assert_eq!(other.mask(), vec![0; 8]);
        assert!(!other.commit_token(1));
        constraint = Constraint::load(constraint.save(), &v).unwrap();
    }
    assert_eq!(module.compile(&v).unwrap().mask_len(), 1);
    let root_b = Grammar::from_expr(Expr::sequence([
        Expr::from(module.clone()),
        Expr::from(TokenSet::new([63])),
    ])).unwrap();
    let policy_b = root_b.compile(&v).unwrap();
    assert_eq!(policy_b.mask_len(), 2);
}

#[test]
fn root_end_ids_are_reserved_and_early_end_rejects_even_with_matching_bytes() {
    let v = vocab();
    let constraint = Grammar::from_expr(Expr::sequence([
        Expr::from(Grammar::from_ebnf(r#"start ::= "a""#)),
        Expr::from(TokenSet::new([7])),
    ])).unwrap().compile(&v).unwrap();
    let mut state = constraint.start();
    assert!(allowed(&state.mask(), 1));
    assert!(!allowed(&state.mask(), 7));
    // An end token before a complete match is a failed commit.
    assert!(!state.commit_token(7));
    assert!(state.mask().iter().all(|word| *word == 0));
    assert!(!state.is_accepting());
    let mut valid = constraint.start();
    assert!(valid.commit_token(1));
    assert!(allowed(&valid.mask(), 7));
    assert!(valid.commit_token(7));
    assert!(valid.is_accepting());
    assert!(valid.mask().iter().all(|word| *word == 0));
    assert!(!valid.commit_token(1));
    assert!(Grammar::from_expr(Expr::sequence([
        Expr::from(Grammar::from_ebnf(r#"start ::= "a""#)),
        Expr::from(TokenSet::new([7, 7])),
    ])).is_ok());
}

#[test]
fn exact_only_model_tokens_survive_open_module_and_constraint_roundtrips() {
    let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]).with_special_tokens([2, 77]);
    let mut open = Module::new(&Grammar::from_glrm(
        r#"glrm 1; start start; extern token CONTROL; nt start = "x" CONTROL "y";"#,
    ));
    open.precompile(&vocab).unwrap();
    let mut loaded_open = Module::load(&open.save()).unwrap();
    loaded_open.bind("CONTROL", TokenSet::from(2)).unwrap();
    let constraint = loaded_open.compile(&vocab).unwrap();

    let mut state = constraint.start();
    assert!(state.commit_token(0));
    assert!(allowed(&state.mask(), 2));
    assert!(!allowed(&state.mask(), 77));
    assert!(state.commit_token(2));
    assert!(state.commit_token(1));
    assert!(state.is_accepting());

    let loaded = Constraint::load(constraint.save(), &vocab).unwrap();
    let mut state = loaded.start();
    assert!(state.commit_token(0));
    assert!(allowed(&state.mask(), 2));
    assert!(state.commit_token(2));
    assert!(state.commit_token(1));
    assert!(state.is_accepting());

    let missing_domain = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
    assert!(Module::load(&open.save()).is_ok());
    assert!(Constraint::load(constraint.save(), &missing_domain).is_err());
}

// -------------------------------------------------------------------------
// 11 Focused New Tests Required by the Spec & Plan
// -------------------------------------------------------------------------

#[test]
fn focused_1_bind_atomicity_on_error() {
    let v = vocab();
    let mut m = Module::new(&Grammar::from_glrm(HOST));
    m.precompile(&v).unwrap();
    let before_save = m.save();
    let before_unbound: Vec<_> = m.externs().unbound().iter().map(|e| e.name().to_string()).collect();

    // 1. Error: unknown extern
    assert!(m.bind("nonexistent", TokenSet::from(7)).is_err());
    assert_eq!(m.save(), before_save);
    assert_eq!(
        m.externs().unbound().iter().map(|e| e.name().to_string()).collect::<Vec<_>>(),
        before_unbound
    );

    // 2. Error: qualified path
    assert!(m.bind("child::sub", TokenSet::from(7)).is_err());
    assert_eq!(m.save(), before_save);

    // 3. Error: open child
    let open_child = Module::new(&Grammar::from_glrm(OPEN));
    assert!(m.bind("child", open_child).is_err());
    assert_eq!(m.save(), before_save);
    assert_eq!(
        m.externs().unbound().iter().map(|e| e.name().to_string()).collect::<Vec<_>>(),
        before_unbound
    );
}

#[test]
fn focused_2_rebinding_error() {
    let mut m = Module::new(&Grammar::from_glrm(HOST));
    let child = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    m.bind("child", child).unwrap();

    let err = m.bind("child", TokenSet::from(7)).unwrap_err().to_string();
    assert!(err.contains("already bound"), "expected 'already bound', got: {err}");
}

#[test]
fn focused_3_qualified_path_rejection() {
    let mut m = Module::new(&Grammar::from_glrm(HOST));
    let err = m.bind("a::b", TokenSet::from(1)).unwrap_err().to_string();
    assert!(err.contains("qualified paths are not supported"), "{err}");
}

#[test]
fn focused_4_open_child_rejection() {
    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    let open_child = Module::new(&Grammar::from_glrm(OPEN));
    let err = parent.bind("child", open_child).unwrap_err().to_string();
    assert!(err.contains("cannot bind \"child\": the value has unbound externs [\"item\"]"), "{err}");

    // Also via Expr
    let open_expr = Expr::sequence([Expr::from(Module::new(&Grammar::from_glrm(OPEN)))]);
    let err_expr = parent.bind("child", open_expr).unwrap_err().to_string();
    assert!(err_expr.contains("unbound externs"), "{err_expr}");
}

#[test]
fn focused_5_externs_get_iter_order_len_unbound() {
    let g = Grammar::from_glrm(
        "glrm 1; start s; extern token b; extern grammar a; extern token c; nt s = a b c;",
    );
    let mut m = Module::new(&g);
    let externs = m.externs();
    assert_eq!(externs.len(), 3);
    assert!(!externs.is_empty());

    // get
    let b_ext = externs.get("b").expect("b must exist");
    assert_eq!(b_ext.name(), "b");
    assert_eq!(b_ext.kind(), ExternKind::Token);

    let a_ext = externs.get("a").expect("a must exist");
    assert_eq!(a_ext.name(), "a");
    assert_eq!(a_ext.kind(), ExternKind::Grammar);

    assert!(externs.get("missing").is_none());

    // iter order must match declared.names
    let names: Vec<String> = externs.iter().map(|e| e.name().to_string()).collect();
    assert_eq!(names, vec!["a", "b", "c"]);

    // unbound before binding
    assert_eq!(externs.unbound().len(), 3);

    // bind one
    m.bind("b", TokenSet::from(7)).unwrap();
    let externs2 = m.externs();
    assert_eq!(externs2.len(), 3);
    assert_eq!(externs2.unbound().len(), 2);
    let unbound_names: Vec<String> = externs2.unbound().iter().map(|e| e.name().to_string()).collect();
    assert_eq!(unbound_names, vec!["a", "c"]);
}

#[test]
fn focused_6_compile_options_never_touch_retained_work() {
    let v = vocab();
    let m = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    assert_eq!(MaskStrategy::FAST_BUILD, MaskStrategy::TokenTrie);
    assert_eq!(MaskStrategy::FAST_RUNTIME, MaskStrategy::ParserDwa);
    assert_eq!(BoundaryMaskStrategy::FAST_BUILD, BoundaryMaskStrategy::TokenTrie);
    assert_eq!(BoundaryMaskStrategy::FAST_RUNTIME, BoundaryMaskStrategy::ParserDwa);

    let mut m = m;
    m.precompile(&v).unwrap();
    let _saved_before = m.save();

    // Options govern the compile without mutating the module or its retained work
    let c = m
        .compile_with(
            &v,
            CompileOptions::default()
                .mask_strategy(MaskStrategy::ParserDwa)
                .boundary_mask_strategy(BoundaryMaskStrategy::ParserDwa),
        )
        .unwrap();
    let s = c.start();
    assert!(allowed(&s.mask(), 1));
}

#[test]
fn focused_7_boundary_level_minimum() {
    let v = vocab();
    let mut m = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    m.precompile_with(
        &v,
        CompileOptions::default().boundary_mask_strategy(BoundaryMaskStrategy::TokenTrie),
    )
    .unwrap();

    // Switching to the explicit static boundary keeps retained work valid.
    m.precompile_with(
        &v,
        CompileOptions::default().boundary_mask_strategy(BoundaryMaskStrategy::ParserDwa),
    )
    .unwrap();

    // Switching back to the token-trie boundary also keeps retained work valid.
    m.precompile_with(
        &v,
        CompileOptions::default().boundary_mask_strategy(BoundaryMaskStrategy::TokenTrie),
    )
    .unwrap();

    let c = m
        .compile_with(
            &v,
            CompileOptions::default().boundary_mask_strategy(BoundaryMaskStrategy::TokenTrie),
        )
        .unwrap();
    let s = c.start();
    assert!(allowed(&s.mask(), 1));
}

#[test]
fn focused_8_static_parts_are_never_downgraded() {
    let v = vocab();
    let mut leaf = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    leaf.precompile_with(&v, CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa))
        .unwrap();

    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    parent.bind("child", leaf).unwrap();

    // TokenTrie requests use the retained static parts instead of downgrading.
    let c = parent
        .compile_with(&v, CompileOptions::default().mask_strategy(MaskStrategy::TokenTrie))
        .unwrap();
    assert!(allowed(&c.start().mask(), 4), "xay must compose statically");
}

#[test]
fn focused_9_auto_reusing_retained_engine() {
    let v = vocab();

    // 1. Auto reuses retained ParserDwa
    let mut m_dwa = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    m_dwa.precompile_with(&v, CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa))
        .unwrap();
    let c_dwa = m_dwa.compile(&v).unwrap();
    let mut s_dwa = c_dwa.start();
    assert!(s_dwa.commit_token(1));
    assert!(s_dwa.is_accepting());

    // 2. Auto reuses retained TokenClassTrie
    let mut m_tct = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    m_tct.precompile_with(&v, CompileOptions::default().mask_strategy(MaskStrategy::TokenClassTrie))
        .unwrap();
    let c_tct = m_tct.compile(&v).unwrap();
    let mut s_tct = c_tct.start();
    assert!(s_tct.commit_token(1));
    assert!(s_tct.is_accepting());

    // 3. Auto never resolves to TokenTrie even if TokenTrie was precompiled
    let mut m_tt = Module::new(&Grammar::from_ebnf(r#"start ::= "a""#));
    m_tt.precompile_with(&v, CompileOptions::default().mask_strategy(MaskStrategy::TokenTrie))
        .unwrap();
    // TokenTrie precompilation retains no work, so Auto builds ParserDwa.
    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    parent.bind("child", m_tt).unwrap();
    let c = parent.compile(&v).unwrap();
    let mut s = c.start();
    assert!(allowed(&s.mask(), 4)); // "xay" token matches host "x" child "a" host "y"
    assert!(s.commit_token(4));
    assert!(s.is_accepting());
}

#[test]
fn focused_10_expr_empty_choice_one_part_literal_token_leaf() {
    let v = vocab();

    // 1. Empty choice matches nothing
    let empty_choice = Expr::choice(Vec::<Expr>::new());
    let mut m_ec = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;"));
    m_ec.bind("E", empty_choice).unwrap();
    let c_ec = m_ec.compile(&v).unwrap();
    assert!(!c_ec.start().is_accepting());
    assert!(c_ec.start().mask().iter().all(|&w| w == 0));
    assert!(!c_ec.start().commit_token(0));

    // 2. One-part choice simplifies to that part
    let single_choice = Expr::choice([Expr::from(Grammar::from_ebnf(r#"start ::= "a""#))]);
    let mut m_sc = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;"));
    m_sc.bind("E", single_choice).unwrap();
    let c_sc = m_sc.compile(&v).unwrap();
    assert!(allowed(&c_sc.start().mask(), 1));

    // 3. Literal matches exact bytes
    let lit = Expr::literal(b"a");
    let mut m_lit = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;"));
    m_lit.bind("E", lit).unwrap();
    let c_lit = m_lit.compile(&v).unwrap();
    assert!(allowed(&c_lit.start().mask(), 1));

    // 4. Token leaf matches exact token ID
    let tok_leaf = Expr::from(TokenSet::from(7));
    let mut m_tok = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;"));
    m_tok.bind("E", tok_leaf).unwrap();
    let c_tok = m_tok.compile(&v).unwrap();
    assert!(allowed(&c_tok.start().mask(), 7));
    assert!(!allowed(&c_tok.start().mask(), 1));
}

#[test]
fn focused_11_grammar_compile_equals_module_new_compile() {
    let v = vocab();
    let grammars = [
        Grammar::from_ebnf(r#"start ::= "a" "b""#),
        Grammar::from_glrm(r#"glrm 1; start s; nt s = "x" "a" "y";"#),
        Grammar::from_json_schema(r#"{"type": "string", "maxLength": 3}"#),
    ];

    for g in grammars {
        let c_grammar = g.compile(&v).unwrap();
        let c_module = Module::new(&g).compile(&v).unwrap();

        assert_eq!(c_grammar.start().mask(), c_module.start().mask());
        assert_eq!(c_grammar.start().is_accepting(), c_module.start().is_accepting());
    }
}

#[test]
fn focused_12_grammar_precompile_returns_a_module() {
    let v = vocab();
    let child = Grammar::from_ebnf(r#"start ::= "a""#);

    // The returned Module is bindable and compiles like Module::new.
    let mut module = Grammar::from_glrm(HOST).precompile(&v).unwrap();
    module.bind("child", child.precompile(&v).unwrap()).unwrap();
    let c = module.compile(&v).unwrap();
    let mut s = c.start();
    assert!(allowed(&s.mask(), 4), "xay must compose the precompiled child");
    assert!(s.commit_token(4));
    assert!(s.is_accepting());

    // Masks equal the explicit Module::new + bind + precompile path.
    let mut reference = Module::new(&Grammar::from_glrm(HOST));
    reference.bind("child", Module::new(&child)).unwrap();
    reference.precompile(&v).unwrap();
    let reference = reference.compile(&v).unwrap();
    assert_eq!(c.start().mask(), reference.start().mask());
    assert_eq!(c.start().is_accepting(), reference.start().is_accepting());

    // Options are honoured by precompile_with.
    let configured = child
        .precompile_with(
            &v,
            CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa),
        )
        .unwrap();
    let plain = Module::new(&child).compile(&v).unwrap();
    assert_eq!(configured.compile(&v).unwrap().start().mask(), plain.start().mask());
}
