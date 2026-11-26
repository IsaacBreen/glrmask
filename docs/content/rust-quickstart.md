```rust
use glrmask::{Grammar, Vocab};

fn main() -> glrmask::Result<()> {
    let grammar = Grammar::from_lark(r#"
        start: "hello" " " "world"
    "#)?;

    // A toy vocabulary of token ID -> bytes. With a real model, build it from the tokenizer.
    let vocab = Vocab::new([
        (0, b"hello".to_vec()), (1, b" ".to_vec()), (2, b"world".to_vec()),
    ])?;

    let constraint = grammar.compile(&vocab)?;
    let mut state = constraint.start();

    // Each mask says which tokens may come next: bit `id` is set if token `id` is allowed.
    assert_eq!(state.mask(), [0b001]); // only "hello"
    assert!(state.commit_token(0));
    assert_eq!(state.mask(), [0b010]); // only " "
    assert!(state.commit_token(1));
    assert!(state.commit_token(2));
    assert!(state.is_accepting());
    Ok(())
}
```
