```rust
# use glrmask::{Grammar, Vocab};
# const EOS: u32 = 256;
# fn model(output: &[u32]) -> Vec<f32> {
#     (0..257).map(|t| ((t * 7919 + output.len() * 104_729) % 1000) as f32).collect()
# }
# fn sample(logits: &[f32], mask: &[u32], accepting: bool) -> u32 {
#     (0..=EOS).filter(|&id| if id == EOS { accepting } else { mask[id as usize / 32] >> (id % 32) & 1 == 1 })
#         .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize])).expect("no allowed token and cannot stop")
# }
# fn main() -> glrmask::Result<()> {
# // Replace with your tokenizer's token bytes
# let token_bytes = (0..256u32).map(|id| (id, vec![id as u8]));
let grammar = Grammar::from_json_schema(r#"
    {
        "type": "object",
        "properties": {
            "color": {
                "enum": [
                    "red", "green", "blue"
                ]
            }
        },
        "required": ["color"],
        "additionalProperties": false
    }
"#)?;

// Token ID -> exact bytes from tokenizer
let vocab = Vocab::new(token_bytes)?;
let constraint = grammar.compile(&vocab)?;
let mut state = constraint.start();

let mut output = Vec::new();
loop {
    let logits = model(&output);
    let mask = state.mask();
    let token = sample(
        &logits, &mask, state.is_accepting()
    );
    if token == EOS { break; }
    assert!(state.commit_token(token));
    output.push(token);
}
# assert!(state.is_accepting());
# let bytes: Vec<u8> = output.into_iter().map(|t| t as u8).collect();
# println!("{}", String::from_utf8_lossy(&bytes));
# Ok(())
# }
```
