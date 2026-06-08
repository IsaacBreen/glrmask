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
fn o4836_pattern_property_key_rejects_partial_unicode_escape_token_like_llguidance_native() {
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

    let mut token_state = constraint.start();
    token_state.commit_bytes(prefix).unwrap();
    assert!(!token_allowed(&token_state.mask(), 0));
    assert!(token_state.commit_token(0).is_err());
}

#[test]
fn string_rejects_partial_unicode_escape_token_like_llguidance_native() {
    let schema = r#"{"type": "string"}"#;
    let token = br#"\uC"#;
    let constraint = Constraint::from_json_schema(schema, &vocab(&[token])).unwrap();

    let mut token_state = constraint.start();
    token_state.commit_bytes(b"\"").unwrap();
    assert!(!token_allowed(&token_state.mask(), 0));
    assert!(token_state.commit_token(0).is_err());
}

#[test]
fn string_mask_allows_bare_unicode_escape_prefix_token_but_rejects_partial_hex_token() {
    let schema = r#"{"type": "string"}"#;
    let bare_unicode_prefix = br#"\u"#;
    let partial_hex_escape = br#"\uC"#;
    let newline_escape = br#"\n"#;
    let constraint = Constraint::from_json_schema(
        schema,
        &vocab(&[bare_unicode_prefix, partial_hex_escape, newline_escape]),
    )
    .unwrap();

    let mut token_state = constraint.start();
    token_state.commit_bytes(b"\"").unwrap();
    let mask = token_state.mask();
    assert!(token_allowed(&mask, 0));
    assert!(!token_allowed(&mask, 1));
    assert!(token_allowed(&mask, 2));
}

#[test]
fn unrestricted_object_key_rejects_partial_unicode_escape_token_like_llguidance_native() {
    let schema = r#"{"type": "object", "additionalProperties": true}"#;
    let prefix = br#"{""#;
    let token = br#"\uC"#;
    let constraint = Constraint::from_json_schema(schema, &vocab(&[token])).unwrap();

    let mut token_state = constraint.start();
    token_state.commit_bytes(prefix).unwrap();
    assert!(!token_allowed(&token_state.mask(), 0));
    assert!(token_state.commit_token(0).is_err());
}
