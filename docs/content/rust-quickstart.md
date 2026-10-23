```rust
use glrmask::{Grammar, Vocab};

fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new(vec![
        (0, b"hello".to_vec()), (1, b" ".to_vec()), (2, b"world".to_vec()),
    ]);
    let grammar = Grammar::from_ebnf(r#"start ::= "hello" " " "world""#)?;
    let constraint = grammar.compile(&vocab)?;
    let mut state = constraint.start();
    for token_id in 0..3 {
        assert_eq!(state.mask()[0], 1 << token_id);
        assert!(state.commit_token(token_id));
    }
    assert!(state.is_accepting());
    Ok(())
}
```
