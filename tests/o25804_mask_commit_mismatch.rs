use glrmask::{Constraint, Vocab};

const QUOTE_TOKEN_ID: u32 = 5;
const QUOTE_TOKEN_BYTES: &[u8] = b"\"";
const SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "a": {
      "type": "string",
      "maxLength": 2
    }
  },
  "required": ["a"],
  "additionalProperties": false
}"#;
const PREFIX: &[u8] = br#"{"a": "x"#;

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    word < mask.len() && ((mask[word] >> bit) & 1) != 0
}

fn minimal_vocab() -> Vocab {
    Vocab::new(
        vec![
            (0, b"{\"".to_vec()),
            (1, b"name".to_vec()),
            (2, b"\":".to_vec()),
            (3, b" \"".to_vec()),
            (4, b"example".to_vec()),
            (QUOTE_TOKEN_ID, QUOTE_TOKEN_BYTES.to_vec()),
        ],
        None,
    )
}

#[test]
fn max_length_string_quote_token_can_be_committed_even_when_mask_omits_it() {
    let constraint = Constraint::from_json_schema(SCHEMA, &minimal_vocab()).unwrap();

    let mut base = constraint.start();
    base.commit_bytes(PREFIX).unwrap();

    let mask_has_quote = token_allowed(&base.mask(), QUOTE_TOKEN_ID);

    let mut commit_token_state = base.clone();
    let commit_token_ok = commit_token_state.commit_token(QUOTE_TOKEN_ID).is_ok();

    let mut commit_bytes_state = base.clone();
    let commit_bytes_ok = commit_bytes_state.commit_bytes(QUOTE_TOKEN_BYTES).is_ok();

    assert!(
        !mask_has_quote && commit_token_ok && commit_bytes_ok,
        "expected crate-level mask/commit mismatch, got mask_has_quote={mask_has_quote}, commit_token_ok={commit_token_ok}, commit_bytes_ok={commit_bytes_ok}, stacks={:?}",
        base.debug_parser_stacks(),
    );
}