use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], token_id: usize) -> bool {
    let word = token_id / 32;
    word < mask.len() && ((mask[word] >> (token_id % 32)) & 1) != 0
}

fn main() {
    // The language { a^n b^n | n >= 1 } is context-free but not regular.
    let grammar = r#"start ::= "a" start "b" | "a" "b""#;
    let vocab = Vocab::new(
        vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
    let constraint = Constraint::from_ebnf(grammar, &vocab).unwrap();
    let mut state = constraint.start();

    // Generate aabb. Before the first b, either another a can open a deeper
    // recursive level or b can begin closing the current one.
    assert!(token_allowed(&state.mask(), 0));
    assert!(!token_allowed(&state.mask(), 1));
    state.commit_token(0).unwrap();

    assert!(token_allowed(&state.mask(), 0));
    assert!(token_allowed(&state.mask(), 1));
    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();

    // Once closing has begun, another a would produce a^2 b a..., which cannot
    // be completed to a string in a^n b^n.
    assert!(!token_allowed(&state.mask(), 0));
    assert!(token_allowed(&state.mask(), 1));
    state.commit_token(1).unwrap();

    assert!(state.is_finished());
    println!("accepted: aabb");
}
