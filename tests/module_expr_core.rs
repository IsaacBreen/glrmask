use glrmask::{CompileOptions, Expr, Grammar, MaskStrategy, Module, Token, Vocab};

fn vocab() -> Vocab {
    Vocab::new_with_exact_token_ids(
        vec![
            (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
            (3, b"y".to_vec()), (4, b"xay".to_vec()), (5, b"ab".to_vec()),
            (6, b"xby".to_vec()), (7, b"a".to_vec()), (8, b"aab".to_vec()),
        ],
        [99],
    )
}

fn allowed(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|w| w & (1 << (id % 32)) != 0)
}

const HOST: &str = "glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";";
const OPEN: &str = "glrm 1; start start; extern grammar item; nt start = item;";

fn a() -> Grammar<'static> {
    Grammar::from_ebnf("start ::= \"a\"")
}

fn b() -> Grammar<'static> {
    Grammar::from_ebnf("start ::= \"b\"")
}

fn accepts(c: &glrmask::Constraint, ids: &[u32]) -> bool {
    let mut s = c.start();
    for &id in ids {
        if s.commit_token(id).is_err() {
            return false;
        }
    }
    s.is_accepting()
}

#[test]
fn prepared_parent_raw_child_and_atomic_snapshots() {
    let v = vocab();
    let g = Grammar::from_glrm(HOST);
    let mut parent = Module::new(&g);
    parent.precompile(&v).unwrap();
    let empty = parent.clone();
    let mut child = Module::new(&Grammar::from_glrm(OPEN));

    // Open child rejection: child has unbound "item"
    let err = parent.bind("child", &child).unwrap_err().to_string();
    assert!(err.contains("unbound externs"), "{err}");

    // Qualified path rejection: "::" is never supported
    let err_qual = parent.bind("child::item", a()).unwrap_err().to_string();
    assert!(err_qual.contains("qualified paths are not supported"), "{err_qual}");

    // Close child first, then bind to parent
    child.bind("item", a()).unwrap();
    parent.bind("child", child).unwrap();
    let bytes = parent.save();

    // Rebinding rejection: "child" is already bound
    assert!(parent.bind("child", b()).is_err());
    assert_eq!(parent.save(), bytes);

    let c = parent.compile(&v).unwrap();
    assert!(accepts(&c, &[4]));
    assert!(!accepts(&c, &[6]));
    assert!(empty.compile(&v).is_err());
}

#[test]
fn options_are_local_and_retained_changes_are_atomic() {
    let v = vocab();
    let mut child = Module::new(&a());
    child.set_mask_strategy(MaskStrategy::TokenClassTrie).unwrap();
    child.precompile(&v).unwrap();
    assert_eq!(child.mask_strategy(), MaskStrategy::TokenClassTrie);
    let _bytes = child.save();

    // Setters store the value and never discard retained work
    child.set_mask_strategy(MaskStrategy::Auto).unwrap();
    assert_eq!(child.mask_strategy(), MaskStrategy::Auto);

    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    parent.bind("child", &child).unwrap();
    parent.set_mask_strategy(MaskStrategy::ParserDwa).unwrap();
    assert!(accepts(&parent.compile(&v).unwrap(), &[4]));
}

#[test]
fn mixed_roundtrip_retains_open_raw_and_prepared_nodes() {
    let v = vocab();
    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    parent.precompile(&v).unwrap();
    let mut child = Module::new(&Grammar::from_glrm(OPEN));
    child.bind("item", a()).unwrap();
    parent.bind("child", child).unwrap();

    let loaded = Module::load(&parent.save(), Some(&v)).unwrap();
    assert!(accepts(&loaded.compile(&v).unwrap(), &[4]));

    let raw = Module::load(&Module::new(&a()).save(), None).unwrap();
    assert!(accepts(&raw.compile(&v).unwrap(), &[1]));

    let mut open = Module::load(&Module::new(&Grammar::from_glrm(OPEN)).save(), None).unwrap();
    open.bind("item", a()).unwrap();
    assert!(accepts(&open.compile(&v).unwrap(), &[1]));
}

#[test]
fn expr_crosses_boundaries_and_parentheses_do_not_change_paths() {
    let v = vocab();
    let union = Expr::choice([Expr::from(a()), Expr::from(b())]);
    let mut host = Module::new(&Grammar::from_glrm(HOST));
    host.bind("child", &union).unwrap();
    let c = host.compile(&v).unwrap();
    assert!(accepts(&c, &[4]));
    assert!(accepts(&c, &[6]));
}

#[test]
fn empty_operators_have_explicit_behavior() {
    let v = vocab();
    // Empty sequence matches epsilon
    let mut host_eps = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;"));
    host_eps.bind("E", Expr::sequence(Vec::<Expr>::new())).unwrap();
    let eps_constraint = host_eps.compile(&v).unwrap();
    assert!(eps_constraint.start().is_accepting());

    // Empty choice matches nothing
    let mut host_choice = Module::new(&Grammar::from_glrm("extern grammar E start s nt s = E;"));
    host_choice.bind("E", Expr::choice(Vec::<Expr>::new())).unwrap();
    let choice_constraint = host_choice.compile(&v).unwrap();
    assert!(!choice_constraint.start().is_accepting());
    assert!(choice_constraint.start().mask().iter().all(|&w| w == 0));
    assert!(choice_constraint.start().commit_token(0).is_err());
}

#[test]
fn exact_events_identity_and_kind_widening_survive_storage() {
    let v = vocab();
    let host = Grammar::from_glrm("glrm 1; start start; extern token mark; nt start = mark;");
    let mut m = Module::new(&host);
    assert!(m.bind("mark", a()).is_err());
    m.bind("mark", v.tokens([7, 99]).unwrap()).unwrap();
    let loaded = Module::load(&m.save(), None).unwrap();
    let c = loaded.compile(&v).unwrap();
    let mask = c.start().mask();
    assert!(allowed(&mask, 7));
    assert!(allowed(&mask, 99));
    assert!(!allowed(&mask, 1));

    let wrong = Vocab::new_with_exact_token_ids(vec![(7, b"different".to_vec())], [99]);
    assert!(Module::load(&m.save(), Some(&wrong)).is_err());
    assert!(loaded.compile(&wrong).is_err());

    let mut wide = Module::new(&Grammar::from_glrm(OPEN));
    wide.bind("item", Token::new(7)).unwrap();
    let c = wide.compile(&v).unwrap();
    assert!(accepts(&c, &[7]));
    assert!(!allowed(&c.start().mask(), 1));
}

#[test]
fn fastbuild_retains_representation_and_mixed_modes_fail_explicitly() {
    let v = vocab();
    let mut leaf = Module::new(&a());
    leaf.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    leaf.precompile(&v).unwrap();
    let leaf = Module::load(&leaf.save(), None).unwrap();
    assert!(accepts(&leaf.compile(&v).unwrap(), &[1]));

    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    parent.set_mask_strategy(MaskStrategy::ParserDwa).unwrap();
    parent.bind("child", leaf).unwrap();
    let e = parent.compile(&v).unwrap_err().to_string();
    assert!(e.contains("cannot combine TOKEN_TRIE parts with PARSER_DWA or TOKEN_CLASS_TRIE parts in one composition"), "{e}");
}

#[test]
fn final_end_tokens_are_root_only() {
    let v = vocab();
    let m = Module::new(&a());
    let c = m.compile_with(&v, CompileOptions::default().end_tokens([99])).unwrap();
    let mut state = c.start();
    assert!(!allowed(&state.mask(), 99));
    state.commit_token(1).unwrap();
    assert!(allowed(&state.mask(), 99));
    state.commit_token(99).unwrap();
}

#[test]
fn malformed_storage_is_rejected() {
    let bytes = Module::new(&a()).save();
    for len in [0, 7, 15, bytes.len() - 1] {
        assert!(Module::load(&bytes[..len], None).is_err());
    }
    let mut bad = bytes.clone();
    bad[7] = b'9';
    assert!(Module::load(&bad, None).is_err());
    let mut bad = bytes;
    bad.push(0);
    assert!(Module::load(&bad, None).is_err());
}

#[test]
fn open_precompilation_and_binding_mismatch_are_atomic() {
    let v = vocab();
    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    parent.precompile(&v).unwrap();
    let original = parent.save();

    let wrong = Vocab::new(vec![(0, b"other".to_vec())]);
    assert!(parent.precompile(&wrong).is_err());
    assert_eq!(parent.save(), original);

    assert!(parent.bind("child", wrong.tokens([0]).unwrap()).is_err());
    assert_eq!(parent.save(), original);

    let mut child = Module::new(&a());
    child.precompile(&wrong).unwrap();
    assert!(parent.bind("child", child).is_err());
    assert_eq!(parent.save(), original);

    let mut child = Module::new(&Grammar::from_glrm(OPEN));
    child.bind("item", Token::new(777)).unwrap();
    assert!(parent.bind("child", child).is_err());
    assert_eq!(parent.save(), original);
}

#[test]
fn raw_module_is_owned_and_can_preserve_deferred_source_errors() {
    let module = {
        let text = String::from("start ::= \"a\"");
        Module::new(&Grammar::from_ebnf(&text))
    };
    assert!(accepts(&Module::load(&module.save(), None).unwrap().compile(&vocab()).unwrap(), &[1]));
    let invalid = Module::new(&Grammar::from_glrm("invalid source"));
    let loaded = Module::load(&invalid.save(), None).unwrap();
    assert!(loaded.compile(&vocab()).is_err());
}

#[test]
fn local_modes_retain_actual_native_or_lr_storage_across_load() {
    for mode in [MaskStrategy::Auto, MaskStrategy::ParserDwa, MaskStrategy::TokenClassTrie, MaskStrategy::TokenTrie] {
        let v = vocab();
        let mut module = Module::new(&a());
        module.set_mask_strategy(mode).unwrap();
        module.precompile(&v).unwrap();
        let loaded = Module::load(&module.save(), Some(&v)).unwrap();
        assert_eq!(loaded.mask_strategy(), module.mask_strategy());
        assert!(accepts(&loaded.compile(&v).unwrap(), &[1]));
    }
}

#[test]
fn homogeneous_fastbuild_open_preparation_and_exact_token_edges() {
    let v = vocab();
    let mut root = Module::new(&Grammar::from_glrm(HOST));
    root.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    root.precompile(&v).unwrap();
    let open = root.save();

    let mut leaf = Module::new(&a());
    leaf.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    leaf.precompile(&v).unwrap();
    root.bind("child", &leaf).unwrap();

    for m in [root.clone(), Module::load(&root.save(), Some(&v)).unwrap()] {
        let c = m.compile(&v).unwrap();
        assert!(accepts(&c, &[4]));
        assert!(!accepts(&c, &[6]));
    }

    let mut token = Module::load(&open, Some(&v)).unwrap();
    token.bind("child", v.tokens([7, 99]).unwrap()).unwrap();
    let c = token.compile(&v).unwrap();
    assert!(accepts(&c, &[0, 7, 3]));
    assert!(!accepts(&c, &[0, 1, 3]));

    let mut terminal = Module::new(&Grammar::from_glrm("glrm 1; start s; extern token mark; nt s = mark;"));
    terminal.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    terminal.precompile(&v).unwrap();
    terminal.bind("mark", Token::new(99)).unwrap();
    let c = Module::load(&terminal.save(), None).unwrap().compile(&v).unwrap();
    assert!(accepts(&c, &[99]));
}

#[test]
fn homogeneous_fastbuild_expression_and_nested_retained_graph() {
    let v = vocab();
    let mut a_mod = Module::new(&a());
    a_mod.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    let mut b_mod = Module::new(&b());
    b_mod.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();

    let expr = Expr::sequence([Expr::from(a_mod), Expr::from(b_mod)]);
    let mut parent = Module::new(&Grammar::from_glrm(HOST));
    parent.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    parent.precompile(&v).unwrap();
    parent.bind("child", expr).unwrap();
    let c = Module::load(&parent.save(), None).unwrap().compile(&v).unwrap();
    assert!(accepts(&c, &[0, 5, 3]));
}

#[test]
fn token_trie_nullable_child_compiles_as_one_grammar() {
    let v = vocab();
    let mut root = Module::new(&Grammar::from_glrm(HOST));
    root.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    let mut child = Module::new(&Grammar::from_ebnf(r#"start ::= "" | "a""#));
    child.set_mask_strategy(MaskStrategy::TokenTrie).unwrap();
    root.bind("child", child).unwrap();
    let c = root.compile(&v).unwrap();
    assert!(accepts(&c, &[0, 3]));
    assert!(accepts(&c, &[0, 1, 3]));
}

#[test]
fn compile_succeeds_on_threads_with_small_stacks() {
    // musl gives new threads 128 KiB stacks by default.
    let worker = std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            let vocab = Vocab::new_with_exact_token_ids(
                vec![(17, b"m17".to_vec()), (29, b"m29".to_vec())],
                [],
            );
            let mut template = Module::new(&Grammar::from_glrm("extern token M start r nt r=M"));
            template.precompile(&vocab).unwrap();
            let mut request = template.clone();
            request.bind("M", Token::new(17)).unwrap();
            request.compile(&vocab).unwrap()
        })
        .unwrap();
    let constraint = worker.join().unwrap();
    assert!(allowed(&constraint.start().mask(), 17));
    assert!(accepts(&constraint, &[17]));
}
