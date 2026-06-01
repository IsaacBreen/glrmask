//! Minimal JSON Schema example for publication docs.

use glrmask::{Constraint, Vocab};

fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new((0u32..=255).map(|b| (b, vec![b as u8])).collect(), None);
    let schema = r#"{
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": false
    }"#;

    let constraint = Constraint::from_json_schema(schema, &vocab)?;
    let mut state = constraint.start();
    state.commit_bytes(br#"{"x":1}"#).unwrap();

    assert!(state.is_finished());
    Ok(())
}
