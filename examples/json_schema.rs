use glrmask::{Constraint, Vocab};

fn byte_vocab() -> Vocab {
    let entries = (0..=255u32).map(|byte| (byte, vec![byte as u8])).collect();
    Vocab::new(entries)
}

fn main() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "ok": { "type": "boolean" }
        },
        "required": ["ok"],
        "additionalProperties": false
    }"#;

    let constraint = Constraint::from_json_schema(schema, &byte_vocab()).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(br#"{"ok": true}"#).unwrap();

    assert!(state.is_finished());
    println!("accepted: {{\"ok\": true}}");
}
