```rust
use glrmask::{Grammar, Vocab};

fn main() -> glrmask::Result<()> {
    let grammar = Grammar::from_json_schema(r#"{
        "type": "object",
        "properties": {"color": {"enum": ["red", "green", "blue"]}},
        "required": ["color"],
        "additionalProperties": false
    }"#)?;

    // One token per byte, plus EOS. With a real model, build this from its tokenizer.
    let vocab = Vocab::new((0..256u32).map(|b| (b, vec![b as u8])))?;
    const EOS: u32 = 256;

    let mut state = grammar.compile(&vocab)?.start();
    let mut output = Vec::new();
    loop {
        let logits = model(&output); // stand-in for your model
        let mask = state.mask();
        let allowed = |t: u32| match t {
            EOS => state.is_accepting(),
            t => mask[t as usize / 32] >> (t % 32) & 1 == 1,
        };
        let token = (0..=EOS)
            .filter(|&t| allowed(t))
            .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]))
            .expect("no allowed token and cannot stop");
        if token == EOS {
            break;
        }
        assert!(state.commit_token(token), "allowed token rejected");
        output.push(token as u8);
    }
    assert!(state.is_accepting(), "incomplete output");
    println!("{}", String::from_utf8_lossy(&output)); // e.g. {"color": "blue"}
    Ok(())
}

fn model(output: &[u8]) -> Vec<f32> {
    (0..257).map(|t| ((t * 7919 + output.len() * 104_729) % 1000) as f32).collect()
}
```
