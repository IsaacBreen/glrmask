use glrmask::{Constraint, Vocab};
use glrmask::__private::ConstraintStateExt as _;

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    word < mask.len() && ((mask[word] >> bit) & 1) != 0
}
const OPENING_TOKEN_ID: u32 = 0;
const OPENING_TOKEN_BYTES: &[u8] = b"{\"";
const QUOTE_TOKEN_ID: u32 = 1;
const QUOTE_TOKEN_BYTES: &[u8] = b"\"";

const SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "maxLength": 32
    }
  },
  "required": ["name"],
  "additionalProperties": false
}"#;

const GLRM_WITNESS: &str = r#"start start;
internal t JSON_STRING_CHAR ::= /(?:[\x20-\x21\x23-\x5B\x5D-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE-\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2}|\\["\\bfnrt]|\\u00(?:[01][0-9A-Fa-f]|7[Ff]))/;
t U32 ::= JSON_STRING_CHAR{0,32};
nt start ::= "{\"name\": \"" U32 "\"";
"#;

const PREFIX: &[u8] = br#"{"name": "example"#;

fn minimized_vocab() -> Vocab {
    Vocab::new(
        vec![
            (OPENING_TOKEN_ID, OPENING_TOKEN_BYTES.to_vec()),
            (QUOTE_TOKEN_ID, QUOTE_TOKEN_BYTES.to_vec()),
        ])
}

fn assert_quote_mask_commit_alignment(constraint: &Constraint) {
    let mut base = constraint.start();
    base.commit_bytes(PREFIX).unwrap();

    let mask_has_quote = token_allowed(&base.mask(), QUOTE_TOKEN_ID);

    let mut commit_token_state = base.clone();
    let commit_token_ok = commit_token_state.commit_token(QUOTE_TOKEN_ID).is_ok();

    let mut commit_bytes_state = base.clone();
    let commit_bytes_ok = commit_bytes_state.commit_bytes(QUOTE_TOKEN_BYTES).is_ok();

    assert!(
        mask_has_quote && commit_token_ok && commit_bytes_ok,
        "expected quote to be mask/commit aligned, got mask_has_quote={mask_has_quote}, commit_token_ok={commit_token_ok}, commit_bytes_ok={commit_bytes_ok}, stacks={:?}",
        base.debug_parser_stacks(),
    );
}

#[test]
fn schema_witness_has_quote_mask_commit_alignment_with_two_token_vocab() {
    let constraint = Constraint::from_json_schema(SCHEMA, &minimized_vocab()).unwrap();
    assert_quote_mask_commit_alignment(&constraint);
}

#[test]
fn glrm_witness_has_quote_mask_commit_alignment_with_two_token_vocab() {
    let constraint = Constraint::from_glrm_grammar(GLRM_WITNESS, &minimized_vocab()).unwrap();
    assert_quote_mask_commit_alignment(&constraint);
}
