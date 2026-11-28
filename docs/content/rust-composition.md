```rust
use glrmask::{Grammar, Module, Vocab};
fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new((0..=255u32).map(|b| (b, vec![b as u8])).collect::<Vec<_>>())?;
    let weather = Grammar::from_json_schema(
        r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}"#,
    )?.compile(&vocab)?;
    let mut call = Module::new(&Grammar::from_glrm(r#"extern grammar ARGS
        start call
        nt call = "<weather>" ARGS "</weather>""#)?);
    call.bind("ARGS", &weather)?;
    let _constraint = call.compile(&vocab)?;
    Ok(())
}
```
