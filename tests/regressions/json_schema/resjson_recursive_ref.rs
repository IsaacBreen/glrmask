//! Reproduce segfault from JsonSchemaStore---resjson schema (recursive $ref).
use glrmask::{Constraint, Vocab};

fn make_vocab(entries: &[&str]) -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
        .collect();
    Vocab::new(entries)
}

#[test]
fn test_resjson_recursive_ref() {
    let vocab = make_vocab(&[
        "{", "}", "\"", ":", ",", " ", "a", "b", "c", "1", "2",
        "\"name\"", "\"value\"", "true", "false", "null",
    ]);

    let schema = r##"{
        "type": "object",
        "additionalProperties": {
            "minProperties": 1,
            "anyOf": [
                {"type": "string"},
                {"$ref": "#/definitions/resource"}
            ]
        },
        "definitions": {
            "resource": {
                "type": "object",
                "additionalProperties": {
                    "minProperties": 1,
                    "anyOf": [
                        {"type": "string"},
                        {"$ref": "#/definitions/resource"}
                    ]
                }
            }
        }
    }"##;

    let result = Constraint::from_json_schema(schema, &vocab);
    // Should either succeed or return an error, NOT segfault
    match result {
        Ok(_) => println!("Compiled successfully"),
        Err(e) => println!("Error (acceptable): {}", e),
    }
}
