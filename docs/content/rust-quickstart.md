```rust
use glrmask::{BuildOptions, Grammar, Optimization, Vocab};

fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new(vec![
        (0, b"\"yes\"".to_vec()),
        (1, b"\"no\"".to_vec()),
        (2, b"<eos>".to_vec()),
    ]);
    let schema = r#"{"type":"string","enum":["yes","no"]}"#;
    let constraint = Grammar::from_json_schema(schema).compile_with(
        &vocab,
        BuildOptions::default()
            .end_tokens([2])
            .optimization(Optimization::Auto),
    )?;

    let mut state = constraint.start();
    let mask = state.mask();
    assert_ne!(mask[0] & 1, 0); // Token 0 is allowed.
    state.commit_token(0)?;
    assert!(state.is_accepting());
    Ok(())
}
```

Rust masks are packed `u32` bitsets. Bit `token_id % 32` of word `token_id / 32` indicates whether that token is allowed.
