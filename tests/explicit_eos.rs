use glrmask::{Constraint, Vocab};

fn token_is_set(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word)
        .is_some_and(|value| value & (1u32 << bit) != 0)
}

#[test]
fn explicit_eos_above_byte_vocab_extends_mask_and_tracks_completion() {
    let eos_token_id = 64;
    let vocab = Vocab::new(vec![(0, b"a".to_vec())], Some(eos_token_id));
    let constraint = Constraint::from_ebnf("start ::= \"a\"", &vocab).unwrap();
    let mut state = constraint.start();

    let initial = state.mask();
    assert_eq!(initial.len(), 3);
    assert!(!token_is_set(&initial, eos_token_id));

    state.commit_token(0).unwrap();
    let complete = state.mask();
    assert!(state.is_complete());
    assert!(token_is_set(&complete, eos_token_id));
}
