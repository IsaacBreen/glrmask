//! Token IDs above `MAX_TOKEN_ID` are errors, not huge allocations.
use glrmask::{Expr, Grammar, Module, TokenSet, Vocab, MAX_TOKEN_ID};

const HUGE_IDS: [u32; 3] = [u32::MAX, 4_000_000_000, MAX_TOKEN_ID + 1];

fn is_invalid_input(error: &glrmask::Error) -> bool {
    error.kind() == glrmask::ErrorKind::InvalidInput
}

fn small_vocab() -> Vocab {
    Vocab::new(vec![(0, b"a".to_vec())])
}

fn token_extern() -> Grammar {
    Grammar::from_glrm("start s extern token E nt s = E").unwrap()
}

#[test]
fn max_token_id_is_two_to_the_24_minus_one() {
    assert_eq!(MAX_TOKEN_ID, 16_777_215);
}

#[test]
fn a_vocab_with_a_huge_id_is_an_error_at_compile() {
    for id in HUGE_IDS {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (id, b"b".to_vec())]);
        let grammar = Grammar::from_glrm(r#"start s nt s = "a""#).unwrap();
        let error = grammar.compile(&vocab).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        assert!(error.to_string().contains(&id.to_string()), "{error}");
        assert!(error.to_string().contains("16777215"), "{error}");
        let error = grammar.precompile(&vocab).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
    }
}

#[test]
fn binding_a_huge_token_id_is_an_error() {
    for id in HUGE_IDS {
        let mut grammar = token_extern();
        let error = grammar.bind("E", id).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        let error = grammar.bind("E", TokenSet::new([1, id])).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");

        let mut module = Module::new(&token_extern());
        let error = module.bind("E", id).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        assert!(module.externs().unbound().get("E").is_some(), "a failed bind binds nothing");
    }
}

#[test]
fn a_huge_token_id_in_an_expression_is_an_error() {
    for id in HUGE_IDS {
        let error = Grammar::from_expr(Expr::sequence([Expr::literal("a"), Expr::from(id)])).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        let mut grammar = Grammar::from_glrm("start s extern grammar G nt s = G").unwrap();
        let error = grammar.bind("G", Expr::from(id)).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
    }
}

#[test]
fn the_maximum_token_id_itself_is_accepted() {
    let mut grammar = token_extern();
    grammar.bind("E", MAX_TOKEN_ID).unwrap();
    let mut grammar = token_extern();
    grammar.bind("E", TokenSet::from_ranges([5..])).unwrap();
    assert_eq!(TokenSet::from_ranges([5..]), TokenSet::from_ranges([5..=MAX_TOKEN_ID]));
}

#[test]
fn ordinary_token_bindings_still_compile() {
    let mut grammar = token_extern();
    grammar.bind("E", 70_000u32).unwrap();
    let constraint = grammar.compile(&small_vocab()).unwrap();
    let mut state = constraint.start();
    assert!(state.commit_token(70_000));
    assert!(state.is_accepting());
}
