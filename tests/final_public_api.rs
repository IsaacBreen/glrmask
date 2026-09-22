use glrmask::{Grammar, Module, Vocab};

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

#[test]
fn immutable_source_and_compiled_bindings_cross_token_boundaries() {
    let v = vocab();
    let parent = Grammar::from_glrm(HOST);
    let child = Grammar::from_ebnf(r#"start ::= "a""#);
    let compiled = child.compile(&v).unwrap();
    for c in [parent.bind("child", &child).unwrap().compile(&v).unwrap(),
              parent.bind("child", &compiled).unwrap().compile(&v).unwrap()] {
        let mut s = c.start();
        assert!(allowed(&s.mask(), 4));
        s.commit_token(4).unwrap();
        assert!(s.is_accepting());
    }
    assert!(parent.compile(&v).is_err());
}

#[test]
fn module_reuse_load_and_closed_link() {
    let v = vocab();
    let host = Grammar::from_glrm(HOST).compile_module(&v).unwrap();
    let a = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    let b = Grammar::from_ebnf(r#"start ::= "b""#).compile_module(&v).unwrap();
    for parent in [&host, &Module::load(host.save()).unwrap()] {
        assert!(parent.link().is_err());
        let bound = parent.bind("child", &a).unwrap().link().unwrap();
        assert!(allowed(&bound.start().mask(), 4));
        let c = parent.bind("child", &b).unwrap().link().unwrap();
        assert!(!allowed(&c.start().mask(), 4));
        assert!(parent.link().is_err());
    }
}

#[test]
fn source_exact_token_identity_is_not_bytes() {
    let v = vocab();
    let c = Grammar::from_glrm(TOKEN_HOST).bind("MARK", v.token(7).unwrap()).unwrap().compile(&v).unwrap();
    let mut s = c.start();
    assert!(!allowed(&s.mask(), 4));
    s.commit_token(0).unwrap();
    assert!(allowed(&s.mask(), 7));
    assert!(!allowed(&s.mask(), 1));
    s.commit_token(7).unwrap();
    s.commit_token(3).unwrap();
    assert!(s.is_accepting());
}

#[test]
fn compiled_exact_token_bind_matches_source_at_every_step() {
    let v = vocab();
    let g = Grammar::from_glrm(TOKEN_HOST);
    let m = g.compile_module(&v).unwrap();
    assert!(m.link().is_err());
    let expected = g.bind("MARK", v.tokens([7, 8]).unwrap()).unwrap().compile(&v).unwrap();
    let actual = m.bind("MARK", v.tokens([7, 8]).unwrap()).unwrap().link().unwrap();
    let mut a = actual.start(); let mut b = expected.start();
    for id in [0, 7, 3] {
        assert_eq!(a.mask(), b.mask(), "before {id}");
        a.commit_token(id).unwrap(); b.commit_token(id).unwrap();
        assert_eq!(a.is_accepting(), b.is_accepting());
    }
    assert_eq!(a.mask(), b.mask());
    assert!(m.link().is_err());
}

#[test]
fn unbound_exact_token_survives_save_load() {
    let v = vocab();
    let m = Grammar::from_glrm(TOKEN_HOST).compile_module(&v).unwrap();
    let loaded = Module::load(m.save()).unwrap();
    assert!(loaded.link().is_err(), "loading must not silently close unresolved token slots");
    let c = loaded.bind("MARK", v.token(7).unwrap()).unwrap().link().unwrap();
    let mut s = c.start();
    s.commit_token(0).unwrap();
    assert!(allowed(&s.mask(), 7));
    assert!(!allowed(&s.mask(), 1));
}

#[test]
fn wrong_vocabulary_and_slot_kind_are_rejected() {
    let v = vocab();
    let other = Vocab::new(vec![(7, b"different".to_vec())]);
    let parent = Grammar::from_glrm(TOKEN_HOST);
    assert!(parent.bind("MARK", other.token(7).unwrap()).unwrap().compile(&v).is_err());
    assert!(Grammar::from_glrm(HOST).bind("child", v.token(7).unwrap()).is_err());
    let child = Grammar::from_ebnf(r#"start ::= "a""#).compile(&v).unwrap();
    assert!(parent.bind("MARK", &child).is_err());
    let incompatible = Grammar::from_ebnf(r#"start ::= "different""#).compile(&other).unwrap();
    assert!(Grammar::from_glrm(HOST).bind("child", &incompatible).unwrap().compile(&v).is_err());
}
