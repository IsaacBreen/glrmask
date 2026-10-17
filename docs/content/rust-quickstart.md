```rust
use glrmask::{Grammar, Vocab};

fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new(vec![
        (0, b"hello".to_vec()), (1, b" ".to_vec()), (2, b"world".to_vec()),
    ]);
    let grammar = Grammar::from_ebnf(r#"start ::= "hello" " " "world""#);
    let constraint = grammar.compile(&vocab)?;
    let mut state = constraint.start();
    for (token_id, expected) in [(0, 1), (1, 2), (2, 4)] {
        assert_eq!(state.mask()[0] & 7, expected);
        state.commit_token(token_id)?;
    }
    assert!(state.is_accepting());
    Ok(())
}
```

Rust masks are packed `u32` bitsets. Bit `token_id % 32` of word `token_id / 32` indicates whether that token is allowed.
