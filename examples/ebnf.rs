use glrmask::{CompileOptions, Constraint, Grammar, Vocab};

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

    let constraint = Constraint::compile(
        Grammar::ebnf(r#"start ::= "hello" " " "world""#),
        &vocab,
        &CompileOptions::default(),
    )
    .unwrap();

    let mut state = constraint.start();
    assert!(token_allowed(&state.mask(), 0));

    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    state.commit_token(2).unwrap();

    assert!(state.is_accepting());
    println!("accepted: hello world");
}
