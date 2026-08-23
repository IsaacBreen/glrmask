use glrmask::{CompileOptions, StaticConstraint as Constraint, Grammar, Vocab};

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

    let vocab = byte_vocab();
    let constraint = Constraint::compile(
        Grammar::json_schema(schema),
        &vocab,
        &CompileOptions::default(),
    )
    .unwrap();
    let mut state = constraint.start();
    state.commit_bytes(br#"{"ok": true}"#).unwrap();

    assert!(state.is_accepting());
    println!("accepted: {{\"ok\": true}}");
}
