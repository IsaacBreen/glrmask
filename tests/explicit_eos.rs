use glrmask::{Constraint, Vocab};

fn allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id % 32;
    word < mask.len() && (mask[word] & (1u32 << bit)) != 0
}

#[test]
fn explicit_eos_above_byte_vocab_is_in_mask_extent() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec())], Some(64));
    let constraint = Constraint::from_ebnf(r#"start ::= "a""#, &vocab).unwrap();
    assert_eq!(constraint.mask_len(), 3);

    let mut state = constraint.start_with_rollback(1);
    let before = state.mask();
    assert!(allowed(&before, 0));
    assert!(!allowed(&before, 64));

    state.commit_token(0).unwrap();
    assert!(state.is_complete());
    let after = state.mask();
    assert!(allowed(&after, 64));

    state.rollback(1).unwrap();
    assert!(!allowed(&state.mask(), 64));
}
