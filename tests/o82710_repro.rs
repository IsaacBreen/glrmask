use glrmask::{Constraint, Vocab};

fn vocab(entries: &[&[u8]]) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec()))
            .collect())
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

fn string_schema() -> &'static str {
    r#"{"type":"string","minLength":0,"maxLength":5000}"#
}

fn object_schema() -> &'static str {
    r#"{
        "type":"object",
        "properties":{
            "aside":{"type":"boolean"},
            "autoplay":{"type":"boolean"},
            "css_class":{"type":"string","pattern":"^[\\w\\s-]+$"},
            "description":{"type":"string","minLength":0,"maxLength":5000}
        },
        "required":[],
        "additionalProperties":true
    }"#
}

fn pattern_whitespace_schema() -> &'static str {
    r#"{"type":"string","pattern":"^[\\w\\s-]+$"}"#
}

fn string_prefix() -> Vec<u8> {
    let mut prefix = String::from("\"");
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    prefix.into_bytes()
}

fn object_prefix() -> Vec<u8> {
    let mut prefix = String::from(
        "{\"aside\": true, \"autoplay\": false, \"css_class\": \"vimeo-video-block\", \"description\": \"",
    );
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    prefix.into_bytes()
}

fn assert_token_in_mask(schema: &str, prefix: &[u8], tokens: &[&[u8]], token_id: usize) {
    let vocab = vocab(tokens);
    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();
    assert!(
        token_allowed(&state.mask(), token_id),
        "token {token_id} should be present in mask"
    );
}

#[test]
fn o82710_string_prefix_allows_disputed_token() {
    assert_token_in_mask(string_schema(), &string_prefix(), &[b"'];?>\"", b" Vimeo"], 0);
}

#[test]
fn o82710_string_prefix_allows_control_token() {
    assert_token_in_mask(string_schema(), &string_prefix(), &[b"');?>\"", b" Vimeo"], 1);
}

#[test]
fn o82710_string_prefix_commits_disputed_single_token() {
    let vocab = vocab(&[b"');?>\""]);
    let constraint = Constraint::from_json_schema(string_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&string_prefix()).unwrap();
    assert!(token_allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
}

#[test]
fn o82710_object_prefix_allows_disputed_token() {
    assert_token_in_mask(object_schema(), &object_prefix(), &[b"');?>\"", b" Vimeo"], 0);
}

#[test]
fn json_schema_pattern_s_allows_ecma_unicode_whitespace_token() {
    let unicode_space = "\u{3000}".as_bytes();
    assert_token_in_mask(pattern_whitespace_schema(), b"\"", &[unicode_space, b"x"], 0);
}

#[test]
fn json_schema_pattern_s_allows_ecma_unicode_whitespace_lead_byte_token() {
    assert_token_in_mask(pattern_whitespace_schema(), b"\"", &[b"\xE2", b"x"], 0);
}

#[test]
fn json_schema_pattern_s_accepts_ecma_unicode_whitespace_string() {
    let vocab = vocab(&["\"".as_bytes(), "\u{3000}".as_bytes()]);
    let constraint = Constraint::from_json_schema(pattern_whitespace_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes("\"\u{3000}\"".as_bytes()).unwrap();
    assert!(state.is_finished());
}

#[test]
fn json_schema_pattern_s_accepts_llguidance_nel_whitespace_string() {
    let vocab = vocab(&["\"".as_bytes(), "\u{0085}".as_bytes()]);
    let constraint = Constraint::from_json_schema(pattern_whitespace_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes("\"\u{0085}\"".as_bytes()).unwrap();
    assert!(state.is_finished());
}
