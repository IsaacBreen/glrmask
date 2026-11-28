```rust
# use glrmask::{Grammar, Module, Vocab};
# fn main() -> glrmask::Result<()> {
# // Replace with your tokenizer's token bytes
# let token_bytes = (0..256u32).map(|id| (id, vec![id as u8]));
// Token ID -> exact bytes from tokenizer
let vocab = Vocab::new(token_bytes)?;

let weather = Grammar::from_json_schema(r#"
    {
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"],
        "additionalProperties": false
    }
"#)?.compile(&vocab)?;

let parent = Grammar::from_glrm(r#"
    extern grammar ARGS
    start call
    nt call = "<weather>" ARGS
              "</weather>"
"#)?;
let mut call = Module::new(&parent);
call.bind("ARGS", &weather)?;
let constraint = call.compile(&vocab)?;
# let mut state = constraint.start();
# assert!(state.commit_bytes(b"<weather>{\"city\": \"Perth\"}</weather>"));
# assert!(state.is_accepting());
# Ok(())
# }
```
