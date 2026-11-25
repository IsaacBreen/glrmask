//! Token IDs above `MAX_TOKEN_ID` are errors, not huge allocations.
use glrmask::{Expr, Grammar, Module, TokenSet, Vocab, MAX_TOKEN_ID};

const HUGE_IDS: [u32; 3] = [u32::MAX, 4_000_000_000, MAX_TOKEN_ID + 1];

fn is_invalid_input(error: &glrmask::Error) -> bool {
    error.kind() == glrmask::ErrorKind::InvalidInput
}

fn small_vocab() -> Vocab {
    Vocab::new(vec![(0, b"a".to_vec())]).unwrap()
}

fn token_extern() -> Grammar {
    Grammar::from_glrm("start s extern token E nt s = E").unwrap()
}

#[test]
fn max_token_id_is_two_to_the_24_minus_one() {
    assert_eq!(MAX_TOKEN_ID, 16_777_215);
}

#[test]
fn a_vocab_with_a_huge_id_is_an_error_at_construction() {
    for id in HUGE_IDS {
        let error = Vocab::new([(0, b"a".as_slice()), (id, b"b".as_slice())]).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        assert!(error.to_string().contains(&id.to_string()), "{error}");
        assert!(error.to_string().contains("16777215"), "{error}");
    }
}

#[test]
fn duplicate_vocab_ids_are_invalid_input() {
    let error = Vocab::new([(7, b"a".as_slice()), (7, b"b".as_slice())]).unwrap_err();
    assert!(is_invalid_input(&error));
    assert_eq!(error.to_string(), "duplicate vocabulary ID 7");
}

#[test]
fn binding_a_huge_token_id_is_an_error() {
    for id in HUGE_IDS {
        let mut grammar = token_extern();
        let error = grammar.bind("E", glrmask::TokenSet::new([id])).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        let error = grammar.bind("E", TokenSet::new([1, id])).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");

        let mut module = Module::new(&token_extern());
        let error = module.bind("E", glrmask::TokenSet::new([id])).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        assert!(module.externs().unbound().get("E").is_some(), "a failed bind binds nothing");
    }
}

#[test]
fn a_huge_token_id_in_an_expression_is_an_error() {
    for id in HUGE_IDS {
        let error = Grammar::from_expr(Expr::sequence([Expr::literal("a"), Expr::from(TokenSet::new([id]))])).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        let mut grammar = Grammar::from_glrm("start s extern grammar G nt s = G").unwrap();
        let error = grammar.bind("G", Expr::from(TokenSet::new([id]))).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
    }
}

#[test]
fn the_maximum_token_id_itself_is_accepted() {
    let mut grammar = token_extern();
    grammar.bind("E", glrmask::TokenSet::new([MAX_TOKEN_ID])).unwrap();
    let mut grammar = token_extern();
    grammar.bind("E", TokenSet::from_ranges(std::iter::once(5..))).unwrap();
    assert_eq!(TokenSet::from_ranges(std::iter::once(5..)), TokenSet::from_ranges(std::iter::once(5..=MAX_TOKEN_ID)));
}

#[test]
fn ordinary_token_bindings_still_compile() {
    let mut grammar = token_extern();
    grammar.bind("E", glrmask::TokenSet::new([70_000u32])).unwrap();
    let constraint = grammar.compile(&small_vocab()).unwrap();
    let mut state = constraint.start();
    assert!(state.commit_token(70_000));
    assert!(state.is_accepting());
}

#[test]
fn a_constructed_above_cap_range_is_rejected_on_use() {
    let tokens = TokenSet::from_ranges(std::iter::once(MAX_TOKEN_ID..=MAX_TOKEN_ID + 1));
    assert!(is_invalid_input(&token_extern().bind("E", tokens.clone()).unwrap_err()));
    assert!(is_invalid_input(&Grammar::from_expr(tokens).unwrap_err()));
}
