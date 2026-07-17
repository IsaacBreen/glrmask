use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], token_id: usize) -> bool {
    let word = token_id / 32;
    word < mask.len() && ((mask[word] >> (token_id % 32)) & 1) != 0
}

fn main() {
    let vocab = Vocab::new(
        vec![
            (0, b"hello".to_vec()),
            (1, b" ".to_vec()),
            (2, b"world".to_vec()),
        ]);

    let constraint = Constraint::from_ebnf(
        r#"start ::= "hello" " " "world""#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    assert!(token_allowed(&state.mask(), 0));

    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    state.commit_token(2).unwrap();

    assert!(state.is_finished());
    println!("accepted: hello world");
}
