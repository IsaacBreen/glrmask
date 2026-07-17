use glrmask::{Constraint, Vocab};

fn byte_vocab() -> Vocab {
    let entries = (0..=255u32).map(|byte| (byte, vec![byte as u8])).collect();
    Vocab::new(entries)
}

#[test]
fn recursive_duplicate_anyof_ref_closed_object_does_not_overflow() {
    let schema = r##"{
        "$schema": "http://json-schema.org/draft-04/schema#",
        "definitions": {
            "M": {
                "type": "object",
                "properties": {
                    "a": {
                        "anyOf": [
                            {"$ref": "#/definitions/M"},
                            {"$ref": "#/definitions/M"}
                        ]
                    }
                },
                "additionalProperties": false
            }
        },
        "$ref": "#/definitions/M"
    }"##;

    let vocab = byte_vocab();
    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"{}").unwrap();
    assert!(state.is_finished());
}

#[test]
fn recursive_mixed_anyof_ref_closed_object_does_not_overflow() {
    let schema = r##"{
        "$schema": "http://json-schema.org/draft-04/schema#",
        "definitions": {
            "A": {
                "type": "object",
                "properties": {
                    "p": {
                        "anyOf": [
                            {"$ref": "#/definitions/A"},
                            {"$ref": "#/definitions/B"}
                        ]
                    }
                },
                "additionalProperties": false
            },
            "B": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        },
        "$ref": "#/definitions/A"
    }"##;

    let vocab = byte_vocab();
    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(b"{}").unwrap();
    assert!(state.is_finished());
}