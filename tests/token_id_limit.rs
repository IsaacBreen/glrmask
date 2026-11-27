//! Token IDs above `MAX_TOKEN_ID` are errors, not huge allocations.
use glrmask::{Grammar, Module, TokenSet, Vocab};

const MAX_TOKEN_ID: u32 = (1 << 24) - 1;

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
        let error = Vocab::new([(0, b"a".to_vec()), (id, b"b".to_vec())]).unwrap_err();
        assert!(is_invalid_input(&error), "{id}: {error}");
        assert!(error.to_string().contains(&id.to_string()), "{error}");
        assert!(error.to_string().contains("supported range"), "{error}");
    }
}

#[test]
fn duplicate_vocab_ids_are_invalid_input() {
    let error = Vocab::new([(7, b"a".to_vec()), (7, b"b".to_vec())]).unwrap_err();
    assert!(is_invalid_input(&error));
    assert_eq!(error.to_string(), "duplicate vocabulary ID 7");
}

#[test]
fn token_sets_panic_at_construction_for_unsupported_ids() {
    for id in HUGE_IDS {
        assert!(std::panic::catch_unwind(|| TokenSet::new([id])).is_err());
        assert!(std::panic::catch_unwind(|| TokenSet::new([1, id])).is_err());
    }
}

#[test]
fn the_maximum_token_id_itself_is_accepted() {
    let mut grammar = Module::new(&token_extern());
    grammar.bind("E", glrmask::TokenSet::new([MAX_TOKEN_ID])).unwrap();
    let mut grammar = Module::new(&token_extern());
    grammar.bind("E", TokenSet::from_ranges(std::iter::once(5..MAX_TOKEN_ID + 1))).unwrap();

}

#[test]
fn ordinary_token_bindings_still_compile() {
    let mut grammar = Module::new(&token_extern());
    grammar.bind("E", glrmask::TokenSet::new([70_000u32])).unwrap();
    let constraint = grammar.compile(&small_vocab()).unwrap();
    let mut state = constraint.start();
    assert!(state.commit_token(70_000));
    assert!(state.is_accepting());
}

#[test]
fn unsupported_nonempty_range_panics_at_construction() {
    assert!(std::panic::catch_unwind(|| TokenSet::from_ranges(std::iter::once(MAX_TOKEN_ID..MAX_TOKEN_ID + 2))).is_err());
    assert_eq!(TokenSet::from_ranges(std::iter::once(u32::MAX..u32::MAX)), TokenSet::new([]));
}
