use glrmask::{Constraint, Vocab};

fn vocab(entries: &[&[u8]]) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec()))
            .collect(),
        None,
    )
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

#[test]
fn o4836_pattern_property_key_allows_partial_unicode_escape_token() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "attributes": {
                "type": "object",
                "patternProperties": {
                    "^.*$": {"type": "array"}
                }
            }
        }
    }"#;
    let prefix = br#"{"attributes": {""#;
    let token = br#"\uC"#;
    let constraint = Constraint::from_json_schema(schema, &vocab(&[token])).unwrap();

    let mut bytes_state = constraint.start();
    bytes_state.commit_bytes(prefix).unwrap();
    bytes_state.commit_bytes(token).unwrap();

    let mut token_state = constraint.start();
    token_state.commit_bytes(prefix).unwrap();
    assert!(token_allowed(&token_state.mask(), 0));
    token_state.commit_token(0).unwrap();
}

#[test]
fn string_allows_partial_unicode_escape_token() {
    let schema = r#"{"type": "string"}"#;
    let token = br#"\uC"#;
    let constraint = Constraint::from_json_schema(schema, &vocab(&[token])).unwrap();

    let mut bytes_state = constraint.start();
    bytes_state.commit_bytes(b"\"").unwrap();
    bytes_state.commit_bytes(token).unwrap();

    let mut token_state = constraint.start();
    token_state.commit_bytes(b"\"").unwrap();
    assert!(token_allowed(&token_state.mask(), 0));
    token_state.commit_token(0).unwrap();
}

#[test]
fn unrestricted_object_key_allows_partial_unicode_escape_token() {
    let schema = r#"{"type": "object", "additionalProperties": true}"#;
    let prefix = br#"{""#;
    let token = br#"\uC"#;
    let constraint = Constraint::from_json_schema(schema, &vocab(&[token])).unwrap();

    let mut bytes_state = constraint.start();
    bytes_state.commit_bytes(prefix).unwrap();
    bytes_state.commit_bytes(token).unwrap();

    let mut token_state = constraint.start();
    token_state.commit_bytes(prefix).unwrap();
    assert!(token_allowed(&token_state.mask(), 0));
    token_state.commit_token(0).unwrap();
}
